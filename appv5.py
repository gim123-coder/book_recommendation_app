#!/usr/bin/env python
# coding: utf-8

# **1 - Install Dependancies**

# In[5]:


# Install dependencies (if you haven't)
# !pip install snowflake-connector-python bcrypt
# !pip install -U langchain-openai


# **2 - Install Dependancies**

# In[7]:


import os
import pandas as pd
import bcrypt
import gradio as gr
import snowflake.connector

import config
import configy

from openai import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# **3 - Load API key using config(y) file**

# In[9]:


# Initialize OpenAI client
client = OpenAI(api_key=configy.OPENAI_AI_KEY)


# **4 - Load Dataset and no book thumbnail placeholder**

# In[11]:


# Load book dataset
books = pd.read_csv("cleaned_books.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"


# **5 - Create Embedding to place vectors for recommendations using 'Vector Search'**

# In[13]:


# Load text descriptions
loader = TextLoader("code_description.txt", encoding="utf-8")
raw_documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separator="\n"
)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings and Chroma vector store
embeddings = OpenAIEmbeddings(openai_api_key=configy.OPENAI_AI_KEY)
db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)


# **6 - Establish Snowflake Connections**

# In[15]:


# Snowflake connection
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=config.SNOWFLAKE_USER,
        password=config.SNOWFLAKE_PASSWORD,
        account=config.SNOWFLAKE_ACCOUNT,
        warehouse=config.SNOWFLAKE_WAREHOUSE,
        database=config.SNOWFLAKE_DATABASE,
        schema=config.SNOWFLAKE_SCHEMA
    )


# **7 - Account Creation and Login Functions**

# In[17]:


# Account creation
def create_account(username, password):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        cs.execute(
            "INSERT INTO USER_ACCOUNTS (username, password_hash) VALUES (%s, %s)",
            (username, hashed_pw)
        )
        conn.commit()
        return "Account created successfully."
    except snowflake.connector.errors.ProgrammingError as e:
        if "unique constraint" in str(e).lower():
            return "Username already exists."
        else:
            return f"Error: {e}"
    finally:
        cs.close()
        conn.close()

# Login
def login(username, password):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    try:
        cs.execute(
            "SELECT password_hash FROM USER_ACCOUNTS WHERE username = %s",
            (username,)
        )
        row = cs.fetchone()
        if row:
            stored_hash = row[0].encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                return True, "Login successful."
            else:
                return False, "Invalid password."
        else:
            return False, "User not found."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        cs.close()
        conn.close()


# **8 - This function retrieves up to 50 vector search results, extracts their ISBNs, filters matching books, and returns the top k recommendations as a DataFrame. (k is a number that represents the relativeness between prompt and book
# )**

# In[19]:


# Retrieve recommendations
def retrieve_recommendations(query: str, top_k: int = 12) -> pd.DataFrame:
    # Retrieve more candidates to improve hit rate
    recs = db_books.similarity_search(query, k=50)

    books_list = []
    for rec in recs:
        try:
            isbn_str = rec.page_content.split()[0].strip("'").strip('"')
            isbn_num = int(isbn_str)
            books_list.append(isbn_num)
        except Exception as e:
            print(f"Skipping invalid record: {e}")

    if not books_list:
        return pd.DataFrame()

    matches = books[books["isbn13"].isin(books_list)].head(top_k)
    print(f"Found {len(matches)} recommendations.")
    return matches


# **9 - This function builds a gallery of book cover images with captions and a dropdown of selectable titles based on the retrieved recommendations.**

# In[21]:


# Recommend books (returning gallery items)
def recommend_books(description, category):
    query_parts = [description]
    if category != "All":
        query_parts.append(f"Category: {category}")
    query = ". ".join(query_parts)

    df = retrieve_recommendations(query, top_k=12)
    if df.empty:
        return [], gr.update(choices=[])

    gallery_data = []
    dropdown_choices = []

    for _, row in df.iterrows():
        img_url = row["large_thumbnail"]

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        desc = " ".join(row["description"].split()[:30]) + "..."
        caption = f"**{row['title']}** by {authors_str}\n\n{desc}"

        gallery_data.append((img_url, caption))
        dropdown_choices.append(f"{row['title']} by {row['authors']} (ISBN: {row['isbn13']})")

    return gallery_data, gr.update(choices=dropdown_choices, value=None)


# **10 - This function inserts the selected book’s ISBN into the user’s reading list table in Snowflake and returns a confirmation message.**

# In[23]:


# Add to reading list
def add_to_reading_list(username, isbn13):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    try:
        cs.execute(
            "INSERT INTO BOOKAPPDB.PUBLIC.USER_READING_LIST (username, isbn13) VALUES (%s, %s)",
            (username, isbn13)
        )
        conn.commit()
        return f"Book {isbn13} added to your reading list."
    except Exception as e:
        return f"Error adding book: {e}"
    finally:
        cs.close()
        conn.close()


# **11 - This function queries the database for all ISBNs saved by the user, retrieves matching book records from the DataFrame, and returns them as a DataFrame.**

# In[25]:


# Retrieve reading list
def get_reading_list(username):
    conn = get_snowflake_connection()
    cs = conn.cursor()
    try:
        cs.execute(
            "SELECT isbn13 FROM BOOKAPPDB.PUBLIC.USER_READING_LIST WHERE username = %s ORDER BY added_at DESC",
            (username,)
        )
        rows = cs.fetchall()
        if not rows:
            return pd.DataFrame()
        isbn_list = [r[0] for r in rows]
        return books[books["isbn13"].isin(isbn_list)]
    except Exception as e:
        print(f"Error retrieving reading list: {e}")
        return pd.DataFrame()
    finally:
        cs.close()
        conn.close()


# **12 - This function formats the user’s reading list as a Markdown string listing each saved book with its title, authors, and ISBN.**

# In[27]:


# Format reading list
def format_reading_list(username):
    df = get_reading_list(username)
    if df.empty:
        return "Your reading list is empty."
    text = "### Your Reading List:\n\n"
    for _, row in df.iterrows():
        text += (
            f"- **{row['title']}** by {row['authors']} (ISBN: {row['isbn13']})\n"
        )
    return text


# **13 - These functions handle UI state updates: login_fn returns visibility controls, a status message, and the username if login succeeds, while signup_fn only returns a message indicating the signup result.**

# In[29]:


# Login function
def login_fn(username, password):
    success, msg = login(username, password)
    if success:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            msg,
            username
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            msg,
            ""
        )

def signup_fn(new_username, new_password, accepted_terms):
    if not accepted_terms:
        return "❌ You must accept the Terms & Conditions before creating an account."
    return create_account(new_username, new_password)


# **14 - This section defines the Gradio UI with login, signup, and app components, sets up all input fields and buttons, and wires up their interactions to the corresponding backend functions to control visibility, display recommendations in a gallery, and manage the user’s reading list.**

# In[56]:


import gradio as gr

# --- Custom CSS for styling ---
css = """
#start-button {
    background-color: orange !important;
    color: white;
    font-size: 18px;
    padding: 10px 30px;
    border-radius: 8px;
    margin-top: 20px;
    border: none;
    transition: background-color 0.3s ease;
}
#start-button:hover {
    background-color: #cc7000 !important;
}
#welcome-text {
    font-size: 16px;
    color: grey;
    text-align: center;
    margin-top: 10px;
}
#terms-checkbox {
    margin-top: 10px;
    font-size: 14px;
}
#terms-accordion button {
    font-weight: bold;
    font-size: 16px;
}
"""

# --- Main Gradio Blocks ---
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="orange")) as demo:
    user_state = gr.State()

    # --- SECTION VISIBILITY ---
    welcome_section = gr.Group(visible=True)
    login_section = gr.Group(visible=False)
    signup_section = gr.Group(visible=False)
    app_section = gr.Group(visible=False)

    # --- WELCOME SECTION ---
    with welcome_section:
        with gr.Row():
            with gr.Column(scale=1): pass
            with gr.Column(scale=2):
                gr.Image("logo_test.png", width=200, show_label=False)
                gr.Markdown("## <center>AI Bookrecommender</center>")
                gr.Markdown(
                    "<center><span id='welcome-text'>"
                    "Welcome to your intelligent reading companion. Discover personalised book suggestions "
                    "powered by AI to match your interests, moods, and reading goals."
                    "</span></center>"
                )
                start_btn = gr.Button("Get Started", elem_id="start-button")
            with gr.Column(scale=1): pass

    # --- LOGIN SECTION ---
    with login_section:
        gr.Markdown("### Login to continue")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_error = gr.Markdown()

    # --- SIGNUP SECTION ---
    with signup_section:
        gr.Markdown("### Create an Account")
        new_username = gr.Textbox(label="New Username")
        new_password = gr.Textbox(label="New Password", type="password")

        with gr.Accordion("View Terms and Conditions", open=False, elem_id="terms-accordion"):
            gr.Markdown("""
Welcome to **AI Bookrecommender**. By creating an account, you agree to the following:

- We use AI to suggest books based on your input.
- You are responsible for the content you submit.
- Your data will not be shared with third parties.
- Book suggestions may not always be accurate or suitable.
- We are not liable for how the app is used.

By checking the box below and clicking "Create Account", you confirm that you accept these terms.
""")

        terms_checkbox = gr.Checkbox(label="I have read and accept the Terms & Conditions", elem_id="terms-checkbox")
        signup_btn = gr.Button("Create Account")
        signup_msg = gr.Markdown()

    # --- APP SECTION ---
    with app_section:
        gr.Markdown("### Book Recommender")
        with gr.Row(equal_height=True):
            description = gr.Textbox(label="Describe the type of book you want")
            category = gr.Dropdown(
                label="Select a category:",
                choices=[
                    "All",
                    "History",
                    "Romance",
                    "Mystery/Thriller",
                    "Science Fiction/Fantasy",
                    "Biography/Memoir",
                    "Self-Help",
                    "Religion",
                    "Science/Technology",
                    "Philosophy",
                    "Poetry",
                    "Art",
                    "Children's",
                    "Other"
                ],
                value="All"
            )
            find_button = gr.Button("Find Recommendations")

        output = gr.Gallery(label="Recommended Books", columns=4, rows=3, allow_preview=True)
        saved_dropdown = gr.Dropdown(
            label="Select a book to save to your reading list",
            choices=[],
            interactive=True
        )
        save_button = gr.Button("Save to Reading List")
        save_status = gr.Markdown()
        view_list_button = gr.Button("View My Reading List")
        reading_list_output = gr.Markdown()

    # --- Button Logic ---
    start_btn.click(
        fn=lambda: (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        ),
        inputs=[],
        outputs=[welcome_section, login_section, signup_section]
    )

    login_btn.click(
        fn=login_fn,
        inputs=[username, password],
        outputs=[login_section, signup_section, app_section, login_error, user_state]
    )

    signup_btn.click(
        fn=lambda new_username, new_password, accepted_terms: (
            "❌ You must accept the Terms & Conditions to create an account."
            if not accepted_terms
            else create_account(new_username, new_password)
        ),
        inputs=[new_username, new_password, terms_checkbox],
        outputs=signup_msg
    )

    find_button.click(
        fn=recommend_books,
        inputs=[description, category],
        outputs=[output, saved_dropdown]
    )

    save_button.click(
        fn=lambda selected_label, username: add_to_reading_list(
            username,
            int(selected_label.split("(ISBN:")[1].strip(") "))
        ),
        inputs=[saved_dropdown, user_state],
        outputs=save_status
    )

    view_list_button.click(
        fn=format_reading_list,
        inputs=user_state,
        outputs=reading_list_output
    )


# **15 - Run the code**

# In[58]:


if __name__ == "__main__":
    demo.launch()


# **Uncomment to close**

# In[35]:


demo.close()


# In[ ]:




