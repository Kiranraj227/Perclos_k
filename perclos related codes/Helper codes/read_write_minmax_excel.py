import pandas as pd
import numpy
import random
import openpyxl
import xlsxwriter

EAR_Avg_max = 0.00
EAR_Avg_min = 0.00
EAR_Avg_smooth_max = 0.00
EAR_Avg_smooth_min = 0.00
EAR_Left_max = 0.00
EAR_Left_min = 0.00
EAR_Right_max = 0.00
EAR_Right_min = 0.00
ec = 1

subject_list = [('01', (570, 50)), ('02', (570, 150)), ('03', (580, 150)), ('04', (570, 180)), ('05', (580, 150)),
                ('06', (600, 150)), ('07', (600, 180)), ('08', (470, 200)), ('10', (470, 220)), ('11', (470, 160)),
                ('13', (470, 160)), ('15', (460, 220)), ('16', (460, 220)), ('17', (500, 110)), ('18', (500, 220)),
                ('19', (470, 180)), ('20', (470, 150)), ('22', (500, 100)), ('23', (500, 150)), ('24', (500, 150)),
                ('26', (500, 200)), ('27', (500, 240))]

EAR_data_df = pd.DataFrame()

ec_list = [1, 0]

print("reading files...")

for subject_number, bb_start in subject_list:

    for ec in ec_list:
        if ec == 1:
            df = pd.read_excel('E:\Perclos_k_io\EC and EO outputs_mp\PERCLOS_DATASET_{}_EC.xlsx'.format(subject_number))

            EAR_Avg_min = df["EAR_avg"].min()
            EAR_Avg_smooth_min = df["EAR_avg_smooth"].min()
            EAR_Left_min = df["EAR_left"].min()
            EAR_Right_min = df["EAR_right"].min()
            print("ec done")
        else:
            df = pd.read_excel('E:\Perclos_k_io\EC and EO outputs_mp\PERCLOS_DATASET_{}_EO.xlsx'.format(subject_number))
            EAR_Avg_max = df["EAR_avg"].max()
            EAR_Avg_smooth_max = df["EAR_avg_smooth"].max()
            EAR_Left_max = df["EAR_left"].max()
            EAR_Right_max = df["EAR_right"].max()
            print("eo done")

    EAR_Avg_threshold = EAR_Avg_min + ((EAR_Avg_max - EAR_Avg_min)*0.2)
    EAR_Avg_smooth_threshold = EAR_Avg_smooth_min + ((EAR_Avg_smooth_max - EAR_Avg_smooth_min) * 0.2)
    EAR_Left_threshold = EAR_Left_min + ((EAR_Left_max - EAR_Left_min) * 0.2)
    EAR_Right_threshold = EAR_Right_min + ((EAR_Right_max - EAR_Right_min) * 0.2)


    EAR_info_dict = {
        'Subject Number': [subject_number],
        'EAR_Avg max EO': [EAR_Avg_max],
        'EAR_Avg min EC': [EAR_Avg_min],
        'EAR_Avg threshold': [EAR_Avg_threshold],
        'EAR_Avg_smooth max EO': [EAR_Avg_smooth_max],
        'EAR_Avg_smooth min EC': [EAR_Avg_smooth_min],
        'EAR_Avg_smooth threshold': [EAR_Avg_smooth_threshold],
        'EAR_Left max EO': [EAR_Left_max],
        'EAR_Left min EC': [EAR_Left_min],
        'EAR_Left threshold': [EAR_Left_threshold],
        'EAR_Right max EO': [EAR_Right_max],
        'EAR_Right min EC': [EAR_Right_min],
        'EAR_Right threshold': [EAR_Right_threshold]
    }

    EAR_info = pd.DataFrame(EAR_info_dict)

    EAR_data_df = pd.concat([EAR_data_df, EAR_info])
    print(f"subject {subject_number} done!")


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('EAR_summary.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
EAR_data_df.to_excel(writer, sheet_name='EAR_threshold', index=False, float_format="%.2f")

# Close the Pandas Excel writer and output the Excel file.
writer.save()
print("all done")
