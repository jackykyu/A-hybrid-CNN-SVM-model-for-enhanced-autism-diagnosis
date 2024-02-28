"timeprocess.py" : Integrate information about functional connectivity through a dictionary
structure
data_object_with_results = {
                'matrix': original_matrix,
                'output_1': staticFC_matrix,
                'output_2': dynamic_matrix_after_PCA,
                'group': group_ID,
                'subid': subid
            }

"hebing.py": Add the information about SRS score
structure
new_npy_data = {
            'matrix': matrix,
            'static': output_1,
            'dynamic': output_2,
            'srs': srs_values,
            'group': group,
            'subid': subid
        }
