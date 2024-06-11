from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

def extract_financial_data(text):
    prompt = get_prompt_financial() + text
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return pd.DataFrame(data.items(), columns=["Measure", "Value"])
    except (json.JSONDecodeError, IndexError):
        pass
    
    return pd.DataFrame({
        "Measure": ["Company Name", "Stock Symbol", "Revenue", "Net Income", "EPS"],
        "Value": ["", "", "", "", ""]
    })

def get_prompt_financial():
    return '''Please retrieve company name, revenue, net income and earnings per share (a.k.a. EPS)
    from the following news article. If you can't find the information from this article 
    then return "". Do not make things up.    
    Then retrieve a stock symbol corresponding to that company. For this you can use
    your general knowledge (it doesn't have to be from this article). Always return your
    response as a valid JSON string. The format of that string should be this, 
    {
        "Company Name": "Walmart",
        "Stock Symbol": "WMT",
        "Revenue": "12.34 million",
        "Net Income": "34.78 million",
        "EPS": "2.1 $"
    }
    News Article:
    ============

    '''

if __name__ == '__main__':
    text = '''
    Tesla recently released its Q1 2024 earnings report, which presented several challenges for the company. Tesla's total revenue for the first quarter was $21.3 billion, falling short of Wall Street's expectations of $22.2 billion. The company's adjusted earnings per share (EPS) were reported at $0.45, also below the anticipated $0.51 per share. This represents a significant year-over-year decline in revenue, dropping from $25.17 billion in Q1 2023【11†source】【12†source】.

    The shortfall in earnings was attributed to a combination of factors, including a reduced average selling price of vehicles, decreased vehicle deliveries, and increased operating expenses. Specifically, Tesla faced production challenges due to the ramp-up of the updated Model 3 at the Fremont factory and disruptions caused by the Red Sea conflict and an arson attack at Gigafactory Berlin【11†source】【12†source】.

    Despite these hurdles, Tesla continues to invest in future growth, including advancements in AI, production capacity, and its Supercharger network. The company reported $2.8 billion in capital expenditures for the quarter. Additionally, Tesla's cash, cash equivalents, and investments totaled $26.9 billion at the end of Q1 2024【11†source】.

    For more details, you can read the full earnings report on [Teslarati](https://www.teslarati.com) and [Shacknews](https://www.shacknews.com).
    '''
    
    df = extract_financial_data(text)
    print(df.to_string())