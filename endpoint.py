import requests
import json

scoring_uri = "http://ee2d7936-a6c9-42d7-8e2c-7e1d00e71433.southcentralus.azurecontainer.io/score"
# # Two sets of data to score, so we get two results back
data = [
    [60, 0,    590,   1,   30,  1, 300000, 2.7, 130, 0, 0, 3],
    [45, 0,    2413,   0,   38,  0, 140000, 1.4, 140, 1, 1, 280]
]
# Convert to JSON string
input_data = json.dumps(data)
# Set the content type
headers = {'Content-Type': 'application/json'}
# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
