import os 

if __name__ == '__main__':
    lr_rate = [0.00215,0.00225,0.00235,0.00245,0.00255,0.00265,0.00275,0.00285,0.00295]
    pssm_weight=[0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
    for lr in lr_rate:
        for weight in pssm_weight:
            print(f"running... lr {lr} weight {weight}")
            cmd = "python tape_model.py --learning_rate {} --pssm_weight {}".format(lr,weight)
            os.system(cmd)

