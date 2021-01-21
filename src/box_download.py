from boxsdk import OAuth2
import argparse

# Use argparse to load in failed data files
parser = argparse.ArgumentParser()
parser.add_argument('--failed_file', default="/home/data/elephants/rawdata/NewLocationData/HHperformance_sounds/failed_files.txt",
     help='File with the failed download files')
parser.add_argument('--failed', action='store_true',
    help='Just try dowloading the previously failed files')

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

# Load in the set of failed files if downloading them
args = parser.parse_args()
failed_files = None
if args.failed:
    failed_files = set()
    f = open(args.failed_file, 'r')
    for line in f:
        failed_files.add(line)

import os
def download(folder, path, failed_files_out, failed_files):
    # TODO: Make folder locally
    path += folder.name + '/'
    os.makedirs(path, exist_ok=True)
    # Open the failed_files_out
    failed = open(path + failed_files_out, 'w')
    items = folder.get_items()
    for item in items:
        print (item.name)
        if item.type == 'file' and (failed_files is None or item.name in failed_files): #and not os.path.exists(path + item.name):
            print ("Downloading")
            output_file = open(path + item.name, 'wb')
            # Let us keep trying to download the file!
            i = 0
            while True:
                try: # Catch some weird exception
                    client.file(file_id=item.id).download_to(output_file)
                except Exception as e:
                    #failed.write(item.name + "\n") 
                    print (item.name + " failed x" + str(i))
                    i += 1
                else:
                    print ("It worked!")
                    break
        else:
            print ("Already exists")

    failed.close()


shared_folder = client.get_shared_item("https://cornell.box.com/s/m286jb2r44nk6dw80urg3xjgggdcq88g") 

download(shared_folder, "/home/data/elephants/rawdata/NewLocationData/", "failed_files.txt", failed_files)

