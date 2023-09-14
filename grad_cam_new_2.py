import numpy as np
from base_cam_new_2 import BaseCAM_New_2


class GradCAM_New_2(BaseCAM_New_2):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM_New_2,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)
        #print('GraCAM newnewne')

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        #print(len(target_category))
        #print(len(grads))
        return np.mean(grads, axis=(2, 3))