# Regarding the remaining issues with competition data
610101 | Evan Calabrese | 2025-10-01T16:33 | votes 23

Hi all,

As many of you have pointed out, there are still a few remaining issues with the competition data. We have been working hard behind the scenes for the past few months to correct a variety of issues that have come up including several that required reprocessing and expert manual review of the data. We are happy that we fixed several of the bigger issues, but we know we haven't been able to fix them all. 

Unfortunately, kaggle's process for updating the competition dataset is extremely slow, taking ~1 week for minor updates such as modifying a single CSV. They have their reasons for this, and I can't comment on the merit of this approach. Given this, and the <2 week timeframe remaining in the challenge, we will NOT be making any additional changes to the dataset.

This project was very ambitious from the start. Not only is it one of the largest ever medical imaging competition datasets ever released, but it also has several other complexities including inhomogeneous multi-site data, nuances in DICOM format, and several different data types (image, categorical, location, segmentation etc). It also required hundreds of hours of expert manual annotation. We are proud of the dataset we have created, but we know it still has many flaws. We sincerely hope you understand.

I am pinning this thread for visibility and in the hopes that we can use it to:

1) Have a single place for aggregating known remaining issues with the dataset.

2) Sharing any code (if you are willing) that competitors can be use to patch, fix, or avoid any remaining dataset issues.

Thanks, and good luck in the final 2 weeks!

## Comments
- [rsjx | 2025-10-07T01:58 | v3] It appears that some participants rely on DICOM attributes during inference, and for certain Multi-Frame DICOMs, these attributes may be missing, leading to invalid or failed inferences. In such cases, you may consider using the following default parameters as a shared function group:
  
  spacing=(0.5, 0.5), origin=(0, 0, 0), orientation=[1, 0, 0, 0, 1, 0], slice_thickness = 5.0
  
  While I cannot confirm that this will improve performance, it may be worth trying.
  - [shiba-inu | 2025-10-07T02:51 | v0] @evancalabrese  . @rachitsaluja  
    
    Yes, as you say, it's important to specify it by default.
    
    On the other hand, I have one request.
    
    If possible, could you provide a few examples of multi-frame DICOM data?
    
    Of course, the pixel data can be provided as dummy data.
    
    Simply by sharing such data, I believe we can significantly reduce our debugging time and number of submissions, allowing us to dedicate more time toward improving accuracy. I think this would benefit not only us competition participants but also you as well.
    - [rsjx | 2025-10-08T00:10 | v1] There is multi-frame DICOM data in the training. Which you can find by search using the following- 
      
      ds.SOPClassUID.name
- [AldoCamargo | 2025-10-14T17:02 | v0] Hi there, I am trying to submit my notebook, but is not allowing me to d o it.
- [Jose Luis Sabogal | 2025-10-05T12:36 | v0] It has truly been very useful to try to unify working methods to obtain images and that is already a great contribution to the work being carried out. Thank you very much.
- [Tom | 2025-10-01T22:26 | v0] @evancalabrese   Thanks for your hard work