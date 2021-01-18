from boxsdk import OAuth2

oauth = OAuth2(
    client_id='7qurgji0m5sek66vgqvlzwtk4mwub8mi',
    client_secret='S6ClpoVyTrb8npnfAL6Uix2CEIA8WeR3',
)

auth_url, csrf_token = oauth.get_authorization_url('https://nikitademir.com')

print(auth_url)

# This then needs to be manually done
from boxsdk import LoggingClient
# The fuck you only have 60 seconds to do this. Seems abusrda
access_token, refresh_token = oauth.authenticate('Un0Zg8ptyWxfetjjI7JL1mvZFBs2iU2Q') # Enter auth code in here from redirect link

# Hoping passing in refresh token here means it gets used
oauth = OAuth2(
    client_id='7qurgji0m5sek66vgqvlzwtk4mwub8mi',
    client_secret='S6ClpoVyTrb8npnfAL6Uix2CEIA8WeR3',
    access_token=access_token,
    refresh_token=refresh_token,
)

client = LoggingClient(oauth)
user = client.user().get()
print ("Beginning")
print('User ID is {0}'.format(user.id))
import os
def download(folder, path):
    # TODO: Make folder locally
    path += folder.name + '/'
    os.makedirs(path, exist_ok=True)
    items = folder.get_items()
    for item in items:
        if item.type == 'file':
            output_file = open(path + item.name, 'wb')
            client.file(file_id=item.id).download_to(output_file)


shared_folder = client.get_shared_item("https://stanford.app.box.com/folder/120406029581?s=m286jb2r44nk6dw80urg3xjgggdcq88g")

download(shared_folder, "/home/data/elephants/rawdata/NewLocationData/")