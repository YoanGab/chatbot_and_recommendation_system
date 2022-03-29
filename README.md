# Chatbot and recommendation system
# AirBnB Chatbot - DIA3

## Members
- Yoan GABISON
- Bastien LEDUC

## Installation

```
> pip install virtualenv    
> virtualenv chatbot_project --python=python3.9  
> source chatbot_project/bin/activate  
> pip install -r requirements.txt
```

## .env file
Create a .env file in the root directory of the project.  
This file contains the following information:  
DISCORD_TOKEN: The token of your Discord bot.  

## Running the chatbot
```> python main.py```


## Description
Our project idea was to create a chatbot that was able to recommend AirBnB rooms to a user. We used a recommendation system based on the user's preferences. The chatbot was able to answer questions about the rooms and the user was able to interact with the chatbot. 
## Bot Info
- Chatbot platform: Discord
- [Chat with bot](https://discord.gg/ACDh263rr4) (open on new tab)
- [Working video of this bot](https://www.youtube.com/watch?v=j9UqlvQvxtg)

### Used API
To complete the dataset, we had to scrape Airbnb. Unfortunately, the API was not available.  
We then scrape directly from the website the data we needed.  

### Used Dataset
We used the [Airbnb New York @ 4.DEC.2021](https://www.kaggle.com/datasets/sirapatsam/airbnb-new-york-4dec2021) dataset for building our recommendation system. The dataset contains information about the rooms available in New York City. Those features are the following:
- id (unique id of the room)
- name (name of the room)
- host_id (id of the host)
- host_name (name of the host)
- neighbourhood_group (neighbourhood group) (ex: "Manhattan")
- neighbourhood (neighbourhood located in the neighbourhood group) (ex: "Upper West Side")
- latitude (latitude of the room)
- longitude (longitude of the room)
- room_type (type of the room) (ex: "Entire home/apt")
- price (price of the room)
- minimum_nights (minimum number of nights the room can be booked)
- number_of_reviews (number of reviews the room has)
- last_review (date of the last review)
- reviews_per_month (number of reviews per month)
- calculated_host_listings_count (number of listings the host has)
- availability_365 (number of days the room is available in the year)
- number_of_reviews_ltm (number of reviews the room has in the last month)
- license (license of the host).

Then, we decided to clean the dataset by removing irrelevant features for our recommendation system, and rows with missing values. It resulted in a dataset with the following features:
- id
- neighbourhood_group
- neighbourhood
- room_type
- price
- minimum_nights
- availability_365

And we added two more features:
- rating (the average rating of the room we got from the website)
- image (the url of the image of the room used by our chatbot to display the room)

## Recommender System
We used a recommendation system based on the user's preferences. As we did not have a dataset of users and their ratings, we chose to use the Content Based Recommender System (CBRS). This recommender system is based on the similarity between items and the user's preferences. To do so, we needed first to create the room profiles (with the features selected above) and the user profile, which will have the same features as the room profiles. 

### Numerical to categorical conversion
To create our room profiles, we had to turn some numerical features into categorical features (price, minimum_nights, availability_365). We created a new feature for each of those features, which are ranges of their values (ex: "price_range" for price). In each "feature_range" column, we look at the original value and associate it to the correct range. For example, if the price of a room is between $100 and $200, we put it into the "100-200" category of "price_range" (same thing for minimum_nights, availability_365). 

### One-hot encoding
Once all the features were converted into categorical features, we had to create one-hot encoding for each of them. This is done by creating a new column for each category of the feature. For example, if a room belongs to the "Entire home/apt" category, we create a new column "entire_home_apt" with a value of 1. If the room belongs to the "Private room" category, we create a new column "private_room" with a value of 1. The same goes for the other categories.

### Normalization
When the features are encoded, we normalise them by multiplying each 1 value by the average rating of the room and divide it by 5, the maximum rating of the room. This is done to make the values of the features comparable.  

### User profile
The features of our user profile will be the same ones used for the rooms, until the one-hot encoding. To create our user profile, the chatbot will suggest rooms based on what the user asked, and the user will be able to rate the room between 0 and 5 stars. All the rooms rated by the user will be added to the user profile. Then, we proceed to the same normalization process, but we use this time the user rating insted of the average room rating. Finally, we create our user vector by calculing the mean of the user profile features.

### Similarity
Then, we use the cosine similarity to calculate the similarity between two items. In this case, we calculate the similarity between the user profile vector and the room profiles. We then sort the rooms by their similarity to the user profile, and we return the top 10 rooms.

### Preference
We used the user's preferences to recommend rooms. We got the user's preferences from the Discord chat. For example, the user can say "I want a room in Manhattan above 100€ with 4 stars" or "I want an appartment in Staten Island between 100 and 200€ for 10 nights". We used the user's preferences to filter the rooms based on what the user asked and then recommend rooms that match the the user's preferences.


## Language Processing
To recognize intents, we made a Neural Network model with NLP thanks to [NLTK](https://www.nltk.org/) and [TensorFlow](https://www.tensorflow.org/) libraries.  
When bot is launched, the chatbot will train thanks to the patterns found in the intents.json file.  
After training, the chatbot will be able to recognize the intents of the user's messages.  
For every message received, the chatbot will first recognize the intent of the message and then loop through the regex of the intent to find the different entities.
Once the entities are found, the chatbot will use the entities to find the correct response.


| Regex    | Example            | Match  |
|----------|--------------------|--------| 
| <code>/.\*cheaper([a-zA-Z\s]*)(?P<max_price>[0-9]+).\*/</code> | I want a room cheaper than 100€. | 100 |
| <code>/.\*min\s\*price(\s\*is)?\s*(?P<min_price>[0-9]+).*/</code> | I want a room with a minimum price of 100€. | 100 |
| <code>/.\*in\s+(?P<neighbourhood>[a-zA-Z]*).\*/</code> | I want a room in Manhattan. | Manhattan |
| <code>/.\*(?P<room_type>home&#124;appartment&#124;hotel&#124;private&#124;shared).*/</code> | I want an appartment. | appartment |
| <code>/.\*(?P<rating>[0-9]+)\sstar(s)?.\*/</code> | I want a room with 4 stars. | 4 |
| <code>/.\*my name is (?P<name>[a-zA-Z]*).\*/</code> | My name is John. | John |




### Intents and Entities

| Intent         | Entities                              |
|----------------|---------------------------------------|
| greeting       |                   -                   |
| goodbye        |                   -                   |
| thank you      |                   -                   |
| help           |                   -                   |
|----------------|---------------------------------------|
|                | max_price, min_price, price,          |
| room           | neighbourhood, room_type,             |
|                | minimum_nights, rating (all optional) |
|----------------|---------------------------------------|
| name           | name (optional)                       |
| be_nice        |                   -                   |
| saved_rooms    |                   -                   |
|----------------|---------------------------------------|


## Scenarios
Please describe all the possible scenarios in your chatbot.
### scenario 1:
| User | Bot                                                     |
|------|---------------------------------------------------------|
| Hi   | Welcome to my chatbot😊                                 |
|      | You can use this chat bot using the following examples: |
|      | Hi<br>How are you?<br>I want a pizza                    |

### scenario 2:
| User              | Bot                                                      |
|-------------------|----------------------------------------------------------|
| I want a pizza    | which kind of pizza do you like?<br> Margherita or Greek |
| Margherita please | ok your order is registered. <br> have a nice day.       |


| User                      | Bot                                                |
|---------------------------|----------------------------------------------------|                      
| I want a pizza margherita | ok your order is registered. <br> have a nice day. |

please enter three examples per each scenario


