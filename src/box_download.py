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
def download(folder, path, failed_files):
    # TODO: Make folder locally
    path += folder.name + '/'
    os.makedirs(path, exist_ok=True)
    # Open the failed_files
    failed = open(path + failed_files, 'w')
    items = folder.get_items()
    for item in items:
        print (item.name)
        if item.type == 'file': #and not os.path.exists(path + item.name):
            print ("Downloading")
            output_file = open(path + item.name, 'wb')
            try: # Catch some weird exception
                client.file(file_id=item.id).download_to(output_file)
            except Exception as e:
               failed.write(item.name + "\n") 
        else:
            print ("Already exists")

    failed.close()


shared_folder = client.get_shared_item("https://cornell.box.com/s/m286jb2r44nk6dw80urg3xjgggdcq88g") 

download(shared_folder, "/home/data/elephants/rawdata/NewLocationData/", "failed_files.txt")