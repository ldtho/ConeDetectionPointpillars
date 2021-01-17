from train_lyft import ensemble_csv_multi,ensemble_wbf

if __name__ == "__main__":
    csv_1 = "../outputs/lyft3d_pred_fhd125_v1.csv" #0.197
    csv_2 = "../outputs/lyft3d_pred_fhd125_v0.csv" #0.193
    csv_3 = "../outputs/lyft3d_pred_fhd100_v0.csv"
    csv_4 = "../outputs/lyft3d_pred_fhd100_v1.csv"
    csv_5 = "../outputs/lyft3d_pred_pp100.csv"
    csv_6 = "../outputs/lyft3d_pred_pp125.csv"
    csv_7 = "../outputs/lyft3d_pred_pp250.csv" #0.193

    # the generated file will be in ../outputs/lyft3d_pred_merge_mutli.csv

    #ensemble_csv_multi([csv_1,csv_2,csv_3,csv_4,csv_5,csv_6,csv_7])  #give publi score 0.218
    ensemble_wbf([csv_1,csv_2,csv_3,csv_4,csv_5,csv_6,csv_7])         #give public score 0.222

