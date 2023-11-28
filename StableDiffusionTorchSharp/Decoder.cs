using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace StableDiffusionTorchSharp
{
    public class AttentionBlockA : Module<Tensor, Tensor>
    {
        private readonly GroupNorm groupnorm;
        private readonly SelfAttention attention;

        public AttentionBlockA(long channels) : base("AttentionBlockA")
        {
            groupnorm = GroupNorm(32, channels);
            attention = new SelfAttention(1, channels);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            var residue = x;
            x = groupnorm.forward(x);
            var n = x.shape[0];
            var c = x.shape[1];
            var h = x.shape[2];
            var w = x.shape[3];
            x = x.view(n, c, h * w);
            x = x.transpose(-1, -2);
            x = attention.forward(x);
            x = x.transpose(-1, -2);
            x = x.view(n, c, h, w);
            x += residue;
            return x.MoveToOuterDisposeScope();
        }
    }

    public class ResidualBlockA : Module<Tensor, Tensor>
    {
        private readonly GroupNorm groupnorm_1;
        private readonly Conv2d conv_1;
        private readonly GroupNorm groupnorm_2;
        private readonly Conv2d conv_2;
        private readonly Module<Tensor, Tensor> residual_layer;

        public ResidualBlockA(long in_channels, long out_channels) : base("ResidualBlockA")
        {
            groupnorm_1 = GroupNorm(32, in_channels);
            conv_1 = Conv2d(in_channels, out_channels, kernelSize: 3, padding: 1);
            groupnorm_2 = GroupNorm(32, out_channels);
            conv_2 = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
            if (in_channels == out_channels)
            {
                residual_layer = nn.Identity();
            }
            else
            {
                residual_layer = Conv2d(in_channels, out_channels, kernelSize: 1);
            }
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            var residue = x;
            x = groupnorm_1.forward(x);
            x = torch.nn.functional.silu(x);
            x = conv_1.forward(x);
            x = groupnorm_2.forward(x);
            x = torch.nn.functional.silu(x);
            x = conv_2.forward(x);
            var output = x + residual_layer.forward(residue);
            return output.MoveToOuterDisposeScope();
        }
    }

    public class Decoder : Sequential
    {
        public Decoder() : base(
            Conv2d(4, 4, kernelSize: 1),
            Conv2d(4, 512, kernelSize: 3, padding: 1),
            new ResidualBlockA(512, 512),
            new AttentionBlockA(512),
            new ResidualBlockA(512, 512),
            new ResidualBlockA(512, 512),
            new ResidualBlockA(512, 512),
            new ResidualBlockA(512, 512),
            Upsample(scale_factor: new double[] { 2, 2 }),
            Conv2d(512, 512, kernelSize: 3, padding: 1),
            new ResidualBlockA(512, 512),
            new ResidualBlockA(512, 512),
            new ResidualBlockA(512, 512),
            Upsample(scale_factor: new double[] { 2, 2 }),
            Conv2d(512, 512, kernelSize: 3, padding: 1),
            new ResidualBlockA(512, 256),
            new ResidualBlockA(256, 256),
            new ResidualBlockA(256, 256),
            Upsample(scale_factor: new double[] { 2, 2 }),
            Conv2d(256, 256, kernelSize: 3, padding: 1),
            new ResidualBlockA(256, 128),
            new ResidualBlockA(128, 128),
            new ResidualBlockA(128, 128),
            GroupNorm(32, 128),
            SiLU(),
            Conv2d(128, 3, kernelSize: 3, padding: 1)
            )
        {
        }

        public override Tensor forward(Tensor x)
        {
            x = x / 0.18215f;
            foreach (var module in children())
            {
                x = ((Module<Tensor, Tensor>)module).forward(x);
            }
            return x;
        }
    }
}
