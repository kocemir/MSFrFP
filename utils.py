
min_value=0.5
max_value=1.5


import os
from datetime import datetime 

def create_path(base_folder, batch_size,epoch, domain,run_count,lr,ps,pt, frft_plus:bool):
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and hour as strings
    date_str = current_datetime.strftime('%Y-%m-%d')
    hour_str = current_datetime.strftime('%H-%M-%S')

    # Join the arguments using the appropriate path separator for the operating system
    path_specs= "{}_{}_bc_{}_ep_{}_dm_{}_rn_{}_lr_{}_pl_{}_ps_{}_pt_{}_.pth".format(date_str,hour_str,batch_size, epoch,domain,run_count,lr,frft_plus,ps,pt)
    path_name = os.path.join(base_folder,path_specs)
    return path_name


