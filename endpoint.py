import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://958689da-273e-46d0-8043-9d5cde91624c.southcentralus.azurecontainer.io/score'

# Two sets of data to score, so we get two results back
data = [
        [ 60, 0,    590,   1,   30,  1, 300000, 2.7, 130, 0, 0, 3]
        ]
# Convert to JSON string
input_data = json.dumps(data)

with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)


