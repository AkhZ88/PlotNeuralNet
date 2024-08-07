from pycore.tikzeng import *

# Define your architecture
arch = [
    to_input('input.jpg', width=4, height=4, name='input'),

    # Convolutional blocks (VGG16-like)
    to_ConvConvRelu(name='cr1', s_filer=256, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2, 2), height=40, depth=40, caption="Block 1"),
    to_Pool(name="p1", offset="(0,0,0)", to="(cr1-east)", width=1, height=35, depth=35, opacity=0.5),

    to_ConvConvRelu(name='cr2', s_filer=128, n_filer=(128, 128), offset="(2,0,0)", to="(p1-east)", width=(4, 4), height=30, depth=30, caption="Block 2"),
    to_Pool(name="p2", offset="(0,0,0)", to="(cr2-east)", width=1, height=28, depth=28, opacity=0.5),

    to_ConvConvRelu(name='cr3', s_filer=64, n_filer=(256, 256), offset="(2,0,0)", to="(p2-east)", width=(8, 8), height=20, depth=20, caption="Block 3"),
    to_Pool(name="p3", offset="(0,0,0)", to="(cr3-east)", width=1, height=15, depth=15, opacity=0.5),

    to_ConvConvRelu(name='cr4', s_filer=32, n_filer=(512, 512), offset="(2,0,0)", to="(p3-east)", width=(16, 16), height=10, depth=10, caption="Block 4"),
    to_Pool(name="p4", offset="(0,0,0)", to="(cr4-east)", width=1, height=8, depth=8, opacity=0.5),

    to_ConvConvRelu(name='cr5', s_filer=16, n_filer=(512, 512), offset="(2,0,0)", to="(p4-east)", width=(16, 16), height=5, depth=5, caption="Block 5"),
    to_Pool(name="p5", offset="(0,0,0)", to="(cr5-east)", width=1, height=4, depth=4, opacity=0.5),

    # Global Pooling
    to_GlobalAveragePool(name='gap', offset="(2,0,0)", to="(p5-east)", width=1, height=1, depth=1, opacity=0.5, caption="GlobalAvgPool"),
    to_GlobalMaxPool(name='gmp', offset="(1,0,0)", to="(gap-east)", width=1, height=1, depth=1, opacity=0.5, caption="GlobalMaxPool"),

    # Fully connected layers
    to_FullyConnected(name='fc1', s_filer=1024, offset="(2,0,0)", to="(gmp-east)", width=1, height=1, depth=25, caption="FC1"),
    to_FullyConnected(name='fc2', s_filer=512, offset="(1,0,0)", to="(fc1-east)", width=1, height=1, depth=25, caption="FC2"),
    to_BatchNorm(name='bn1', offset="(0,0,0)", to="(fc2-east)", width=1, height=1, depth=25, opacity=0.5, caption="BatchNorm"),
    to_FullyConnected(name='fc3', s_filer=38, offset="(1,0,0)", to="(bn1-east)", width=1, height=1, depth=10, caption="Output"),

    # SoftMax Layer
    to_SoftMax(name='soft1', n_filer=38, offset="(1,0,0)", to="(fc3-east)", width=1, height=1, depth=10, caption="SoftMax"),
    to_end
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()