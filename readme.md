# ReadMe
* MMCL extends CCL loss
* It re-uses RecZoo/Simplex code available at https://github.com/reczoo/RecZoo/tree/main/matching/cf/SimpleX

## Datasets:
* Check: https://github.com/reczoo/RecZoo/tree/main/matching/cf/SimpleX/data
* For data preparation, also read: https://github.com/reczoo/RecZoo/blob/main/matching/cf/SimpleX/README.md
    * Example: 
    ```
    # convert data format
        cd data/Amazon/AmazonBeauty_m1
        python convert_amazonbeauty_m1.py
        ...
    ```

## Loss Functions:
* RecZoo/Simplex uses another repository for loss functions, namely RecBox
* RecBox is available at https://github.com/reczoo/RecBox
* In MMCL work,
    * RecBox is updated with new loss function
    * An installable version is created by calling the script at `mmcl/recbox_lib.sh`
    * Then, the `RecBox-0.0.4mg` is installed using
     `! pip install <any-path-like-ColabNotebooks>/simplex/RecBox-0.0.4mg`
     Check `mmcl/example_run_loss_functions_yelp.ipynb` as an example.
* Note that only updated files/folders are shared

## Exmple runs
* Check `mmcl/example_run_loss_functions_yelp.ipynb` and `mmcl/example_run_loss_functions_gowalla.ipynb`
