import argparse
import pickle 
import torch


if __name__ == "__main__":
        
    convnext_checkpoint = torch.load('convnext_large_22k_1k_224.pth', map_location='cpu')
    print("Load ckpt from %s" % 'convnext_large_22k_1k_224_Detectron.pth')
    temp_checkpoint_model = {}
    for k,v in convnext_checkpoint.items():
        for k_2,v_2 in v.items():
            temp_checkpoint_model['backbone.bottom_up.'+k_2] =v[k_2] 
    
    print("After updating ")
    for k,v in temp_checkpoint_model.items():
        print(k)
    
    final_check_point = {'model':temp_checkpoint_model}
    print("Final check point is ",final_check_point)

    torch.save(final_check_point, 'convnext_large_22k_1k_224_Detectron.pth')
    
    