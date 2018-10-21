import os
import re


def get_folder(details_path, number):
    details_list = os.listdir(details_path)
    new = False
    if number:
        detail_folder = f"model_{number}/"
        if detail_folder[:-1] not in details_list:
            new = True
    else:
        new = True

    if new:
        if len(details_list):
            detail_folder = sorted(details_list, reverse=True)[0]
            last_no = re.search("_[0-9]+", detail_folder)[0][1:]
            detail_folder = f"model_{int(last_no) + 1}/"
        else:
            detail_folder = "model_1/"
        os.mkdir(details_path + detail_folder)

    return detail_folder, new


def save_details(details_dic, details_path, number=None):

    detail_folder, new = get_folder(details_path, number)
    # Save details
    if new:
        mode = "w"
    else:
        mode = "a"
    with open(details_path + detail_folder + "details.txt", mode) as file:
        for detail, value in details_dic.items():
            file.write(f"{detail}: {value}\n")
