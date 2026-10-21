# modules
# -----------------------------------------------------------------------------------------------------
from cfenv import AppEnv
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

def getDestination(sDestinationService, sDestinationName):

# Check if running locally (for testing)
# -----------------------------------------------------------------------------------------------------
  env = AppEnv()
  dest_service = env.get_service(name=sDestinationService)
  
  if dest_service is None:
    # Running locally - use Key.json for OAuth2 credentials
    print("⚠️  Running in local mode. Using Key.json credentials...")
    
    try:
      # Load Key.json
      # key_file_path = os.path.join(os.path.dirname(__file__), 'Key.json')
      
      # if not os.path.exists(key_file_path):
      #   raise FileNotFoundError(f"Key.json not found at {key_file_path}")
      
      # with open(key_file_path, 'r') as f:
      #   key_data = json.load(f)
      
      # Extract OAuth2 credentials
      client_id = os.getenv('clientid')
      client_secret = os.getenv('clientsecret')
      token_url = os.getenv('token_url') + '/oauth/token'
      
      print(f"✓ Loaded credentials from Key.json")
      print(f"  Token URL: {token_url}")
      
      # Get OAuth2 token using Client Credentials flow
      token_response = requests.post(
        token_url,
        data={
          'grant_type': 'client_credentials',
          'client_id': client_id,
          'client_secret': client_secret
        },
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
      )
      
      if token_response.status_code != 200:
        raise Exception(f"Failed to get token: {token_response.text}")
      
      access_token = token_response.json()['access_token']
      print(f"✓ OAuth2 token obtained successfully")
      
      # Return destination object with auth token
      # URL will be set from environment variable in the Flask app
      return {
        'destinationConfiguration': {
          'URL': os.getenv('WAREHOUSE_ODATA_URL', 'https://warehouse-demo.example.com/odata/v4/warehouse/')
        },
        'authTokens': [
          {
            'http_header': {
              'value': f'Bearer {access_token}'
            }
          }
        ]
      }
      
    except Exception as e:
      print(f"❌ Error in local mode: {str(e)}")
      raise EnvironmentError(f"Failed to authenticate: {str(e)}")

  # Cloud Foundry environment - get credentials and token
  sUaaCredentials = dest_service.credentials["clientid"] + ':' + dest_service.credentials["clientsecret"]

# Request a JWT token to access the destination service
# -----------------------------------------------------------------------------------------------------
  headers = {'Authorization': 'Basic '+sUaaCredentials, 'content-type': 'application/x-www-form-urlencoded'}
  form = [('client_id', dest_service.credentials["clientid"] ),('client_secret', dest_service.credentials["clientsecret"] ), ('grant_type', 'client_credentials')]
  
  url = dest_service.credentials['url'] +"/oauth/token"
  headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
  }

  response = requests.request("POST", url, headers=headers, data=form)

# Search your destination in the destination service
# -----------------------------------------------------------------------------------------------------
  token = response.json()["access_token"]
  headers= { 'Authorization': 'Bearer ' + token }

  r = requests.get(dest_service.credentials["uri"] + '/destination-configuration/v1/destinations/'+sDestinationName, headers=headers)
  print("DEST URI:",dest_service.credentials['uri'])

# Access the destination securely  
# -----------------------------------------------------------------------------------------------------  
  destination = r.json()

  return destination
