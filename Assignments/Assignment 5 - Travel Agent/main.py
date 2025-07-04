import os
from dotenv import load_dotenv
import requests
from langchain.tools import tool
from langchain_groq import ChatGroq
load_dotenv()
# Initialize the model
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Load the API key from environment variables
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
if not GEOAPIFY_API_KEY:
    raise ValueError("API key not found. Please set GEOAPIFY_API_KEY in your environment.")

# Define the agent functions using the @tool decorator
@tool
def attraction_agent(city: str) -> str:
    """
    Agent to search for top tourist attractions in a given city using Geoapify.

    Args:
        city (str): The city where the user wants to find tourist attractions.

    Returns:
        str: A list of top attractions in the specified city.
    """
    try:
        # Step 1: Geocode the city to get coordinates
        geo_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={GEOAPIFY_API_KEY}"
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        features = geo_data.get("features", [])
        if not features:
            return f"Could not find location for '{city}'."

        lon = features[0]["geometry"]["coordinates"][0]
        lat = features[0]["geometry"]["coordinates"][1]

        # Step 2: Query top tourist attractions using tourism category
        places_url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories=tourism.sights,entertainment,tourism.attraction&limit=10&bias=proximity:{lon},{lat}&"
            f"filter=circle:{lon},{lat},5000&apiKey={GEOAPIFY_API_KEY}"
        )
        places_response = requests.get(places_url)
        places_data = places_response.json()
        results = places_data.get("features", [])

        if not results:
            return f"No tourist attractions found in {city}."

        # Format results
        top_attractions = []
        for place in results:
            props = place["properties"]
            name = props.get("name", "Unnamed Attraction")
            addr = props.get("formatted", "No address available")
            top_attractions.append(f"{name} - {addr}")

        return "\n".join(top_attractions)

    except Exception as e:
        return f"An error occurred: {str(e)}"

@tool
def restaurant_agent(city: str) -> str:
    """
    Agent to search for best restaurants in a given city.

    Args:
        city (str): The city where the user wants to find restaurants.
    
    Returns:
        str: A response containing the best restaurants in the specified city.
    """
    
    try:
        # Step 1: Geocode the city to get coordinates
        geo_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={GEOAPIFY_API_KEY}"
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        features = geo_data.get("features", [])
        if not features:
            return f"Could not find location for '{city}'."

        lon = features[0]["geometry"]["coordinates"][0]
        lat = features[0]["geometry"]["coordinates"][1]

        # Step 2: Search restaurants near the city center
        places_url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories=catering.restaurant&limit=10&bias=proximity:{lon},{lat}&"
            f"filter=circle:{lon},{lat},5000&apiKey={GEOAPIFY_API_KEY}"
        )
        places_response = requests.get(places_url)
        places_data = places_response.json()
        results = places_data.get("features", [])

        if not results:
            return f"No restaurants found in {city}."

        # Format results
        top_restaurants = []
        for place in results:
            props = place["properties"]
            name = props.get("name", "Unnamed Restaurant")
            addr = props.get("formatted", "No address available")
            top_restaurants.append(f"{name} - {addr}")
            # print(f"Found restaurant: {name} at {addr}")

        return "\n".join(top_restaurants)

    except Exception as e:
        return f"An error occurred: {str(e)}"


# @tool
# def transportation_agent():

#     pass

# @tool
# def weather_agent():
#     pass

# @tool
# def hotel_agent():
#     pass

# @tool
# def currency_agent():
#     pass

# @tool
# def total_expense_agent():
#     pass

# @tool
# def summary_agent():
#     pass


print(attraction_agent("Mumbai"))
