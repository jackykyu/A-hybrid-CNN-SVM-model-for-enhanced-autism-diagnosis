# A-hybrid-CNN-SVM-model-for-enhanced-autism-diagnosis
****
## Dependencies

List the main dependencies required to run the project along with their versions, including installation methods.<br />
matplotlib==3.7.1<br />
numpy==1.21.5<br />
numpy==1.22.4<br />
pandas==1.5.3<br />
scikit_learn==1.0.2<br />
shap==0.41.0<br />
torch==1.11.0<br />
You can install the dependencies using the following command:
```
pip install -r requirements.txt
```

****
## Dataset
The original data are available at https://fcon_1000.projects.nitrc.org/indi/abide/. The data are preprocessed by tools Dparsf (https://rfmri.org/DPABI). The required parameters are provided in the file named "collecting data.xlsx". And the file "bni.mat" is given as a sample. Folder "infromation of subjects" integrates some information about subjects collected
from https://fcon_1000.projects.nitrc.org/indi/abide/.

****
## Experiment
Folder "datapreprocess" gives the codes used to generate dictionary data for subsequent experiments. The structure of each subject is as follows
```
{
            'matrix': functional connectivity_matrix,
            'static': static_FC,
            'dynamic': dynamic_FC,
            'srs': srs_values,
            'group': group,
            'subid': subid
        }
```
Folder "model" gives the codes for the model mentioned in the article.

****
## License 


