from unittest.mock import patch, MagicMock
import appv5


# === Snowflake Connection ===


def test_get_snowflake_connection():
    with patch("appv5.snowflake.connector.connect") as mock_connect:
        mock_connect.return_value = MagicMock()
        conn = appv5.get_snowflake_connection()
        assert conn is not None
        mock_connect.assert_called_once()


# === Authentication ===


def test_create_account():
    hashed_pw = b"hashed"
    with patch("appv5.bcrypt.hashpw", return_value=hashed_pw), patch(
        "appv5.get_snowflake_connection"
    ) as mock_conn:
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        appv5.create_account("testuser", "password123")
        mock_cursor.execute.assert_called()


def test_login_success():
    hashed_pw = b"hashed"
    with patch("appv5.bcrypt.checkpw", return_value=True), patch(
        "appv5.get_snowflake_connection"
    ) as mock_conn:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "testuser",
            hashed_pw,
        )
        mock_conn.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        assert appv5.login("testuser", "password123") is True


def test_login_failure():
    with patch("appv5.bcrypt.checkpw", return_value=False), patch(
        "appv5.get_snowflake_connection"
    ) as mock_conn:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            "testuser",
            b"wronghash",
        )
        mock_conn.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        assert appv5.login("testuser", "wrongpass") is False


# === Recommendation & Vector Store ===


def test_retrieve_recommendations():
    with patch("appv5.db_books.similarity_search") as mock_search:
        mock_search.return_value = [{"page_content": "Book A"}]
        results = appv5.retrieve_recommendations("AI")
        assert isinstance(results, list)


def test_recommend_books():
    with patch(
        "appv5.retrieve_recommendations",
        return_value=[{"page_content": "Book A"}],
    ):
        books = appv5.recommend_books("Science")
        assert isinstance(books, list)


# === Reading List ===


def test_add_to_reading_list():
    book_info = {"title": "1984", "authors": "George Orwell"}
    with patch("appv5.get_snowflake_connection") as mock_conn:
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        appv5.add_to_reading_list("user1", book_info)
        mock_cursor.execute.assert_called()


def test_get_reading_list():
    with patch("appv5.get_snowflake_connection") as mock_conn:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("Title", "Author", "Image")
        ]
        mock_conn.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        result = appv5.get_reading_list("user1")
        assert isinstance(result, list)


def test_format_reading_list():
    reading_list = [("Book A", "Author A", "image.jpg")]
    result = appv5.format_reading_list(reading_list)
    assert "Book A" in result


# === Gradio Wrapper Functions ===


def test_login_fn_success():
    with patch("appv5.login", return_value=True):
        response = appv5.login_fn("testuser", "password123")
        assert "Welcome" in response


def test_login_fn_failure():
    with patch("appv5.login", return_value=False):
        response = appv5.login_fn("testuser", "wrongpass")
        assert "Invalid" in response


def test_signup_fn():
    with patch("appv5.create_account") as mock_create:
        response = appv5.signup_fn("newuser", "securepass")
        mock_create.assert_called_once()
        assert "account has been created" in response
