using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace StableDiffusionTorchSharp
{
    public class TimeEmbedding : Module<Tensor, Tensor>
    {
        private readonly Linear linear_1;
        private readonly Linear linear_2;

        public TimeEmbedding(long n_embd) : base("TimeEmbedding")
        {
            linear_1 = Linear(n_embd, 4 * n_embd);
            linear_2 = Linear(4 * n_embd, 4 * n_embd);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            x = linear_1.forward(x);
            x = torch.nn.functional.silu(x);
            x = linear_2.forward(x);
            return x.MoveToOuterDisposeScope();
        }
    }

    public class ResidualBlock : Module<Tensor, Tensor, Tensor>
    {
        private readonly GroupNorm groupnorm_feature;
        private readonly Conv2d conv_feature;
        private readonly Linear linear_time;
        private readonly GroupNorm groupnorm_merged;
        private readonly Conv2d conv_merged;
        private readonly Module<Tensor, Tensor> residual_layer;

        public ResidualBlock(long in_channels, long out_channels, long n_time = 1280) : base("ResidualBlock")
        {
            groupnorm_feature = GroupNorm(32, in_channels);
            conv_feature = Conv2d(in_channels, out_channels, kernelSize: 3, padding: 1);
            linear_time = Linear(n_time, out_channels);
            groupnorm_merged = GroupNorm(32, out_channels);
            conv_merged = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
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

        public override Tensor forward(Tensor feature, Tensor time)
        {
            using var _ = NewDisposeScope();
            var residue = feature;
            feature = groupnorm_feature.forward(feature);
            feature = torch.nn.functional.silu(feature);
            feature = conv_feature.forward(feature);

            time = torch.nn.functional.silu(time);
            time = linear_time.forward(time);

            var merged = feature + time.unsqueeze(-1).unsqueeze(-1);
            merged = groupnorm_merged.forward(merged);
            merged = torch.nn.functional.silu(merged);
            merged = conv_merged.forward(merged);

            var output = merged + residual_layer.forward(residue);
            return output.MoveToOuterDisposeScope();
        }
    }

    public class AttentionBlock : Module<Tensor, Tensor, Tensor>
    {
        private readonly GroupNorm groupnorm;
        private readonly Conv2d conv_input;
        private readonly LayerNorm layernorm_1;
        private readonly SelfAttention attention_1;
        private readonly LayerNorm layernorm_2;
        private readonly CrossAttention attention_2;
        private readonly LayerNorm layernorm_3;
        private readonly Linear linear_geglu_1;
        private readonly Linear linear_geglu_2;
        private readonly Conv2d conv_output;

        public AttentionBlock(long n_head, long n_embd, long d_context = 768) : base("AttentionBlock")
        {
            var channels = n_head * n_embd;
            groupnorm = GroupNorm(32, channels);
            conv_input = Conv2d(channels, channels, kernelSize: 1);
            layernorm_1 = LayerNorm(channels);
            attention_1 = new SelfAttention(n_head, channels, in_proj_bias: false);
            layernorm_2 = LayerNorm(channels);
            attention_2 = new CrossAttention(n_head, channels, d_context, in_proj_bias: false);
            layernorm_3 = LayerNorm(channels);
            linear_geglu_1 = Linear(channels, 4 * channels * 2);
            linear_geglu_2 = Linear(4 * channels, channels);
            conv_output = Conv2d(channels, channels, kernelSize: 1);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x, Tensor context)
        {
            using var _ = NewDisposeScope();
            var residue_long = x;
            x = groupnorm.forward(x);
            x = conv_input.forward(x);
            var n = x.shape[0];
            var c = x.shape[1];
            var h = x.shape[2];
            var w = x.shape[3];
            x = x.view(new long[] { n, c, h * w });
            x = x.transpose(-1, -2);
            var residue_short = x;
            x = layernorm_1.forward(x);
            x = attention_1.forward(x);
            x += residue_short;
            residue_short = x;
            x = layernorm_2.forward(x);
            x = attention_2.forward(x, context);
            x += residue_short;
            residue_short = x;
            x = layernorm_3.forward(x);
            var ret = linear_geglu_1.forward(x).chunk(2, -1);
            x = ret[0];
            var gate = ret[1];
            x = x * torch.nn.functional.gelu(gate);
            x = linear_geglu_2.forward(x);
            x += residue_short;
            x = x.transpose(-1, -2);
            x = x.view(new long[] { n, c, h, w });
            var output = conv_output.forward(x) + residue_long;
            return output.MoveToOuterDisposeScope();
        }
    }

    public class Upsample : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv;
        public Upsample(long channels) : base("Upsample")
        {
            conv = Conv2d(channels, channels, kernelSize: 3, padding: 1);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            x = torch.nn.functional.interpolate(x, scale_factor: new double[] { 2, 2 });
            var output = conv.forward(x);
            return output.MoveToOuterDisposeScope();
        }
    }

    class SwitchSequential : Sequential<Tensor, Tensor, Tensor, Tensor>
    {
        internal SwitchSequential(params (string name, torch.nn.Module)[] modules) : base(modules)
        {
        }

        internal SwitchSequential(params torch.nn.Module[] modules) : base(modules)
        {
        }

        public override torch.Tensor forward(torch.Tensor x, torch.Tensor context, torch.Tensor time)
        {
            using var _ = torch.NewDisposeScope();
            foreach (var layer in children())
            {
                switch (layer)
                {
                    case AttentionBlock abl:
                        x = abl.call(x, context);
                        break;
                    case ResidualBlock rbl:
                        x = rbl.call(x, time);
                        break;
                    case torch.nn.Module<torch.Tensor, torch.Tensor> m:
                        x = m.call(x);
                        break;
                }
            }
            return x.MoveToOuterDisposeScope();
        }
    }

    public class UNet : Module<Tensor, Tensor, Tensor, Tensor>
    {
        private readonly ModuleList<SwitchSequential> encoders;
        private readonly SwitchSequential bottleneck;
        private readonly ModuleList<SwitchSequential> decoders;

        public UNet() : base("UNet")
        {
            encoders = new ModuleList<SwitchSequential>(
                new SwitchSequential(Conv2d(4, 320, 3, padding: 1)),
                new SwitchSequential(new ResidualBlock(320, 320), new AttentionBlock(8, 40)),
                new SwitchSequential(new ResidualBlock(320, 320), new AttentionBlock(8, 40)),
                new SwitchSequential(Conv2d(320, 320, 3, stride: 2, padding: 1)),
                new SwitchSequential(new ResidualBlock(320, 640), new AttentionBlock(8, 80)),
                new SwitchSequential(new ResidualBlock(640, 640), new AttentionBlock(8, 80)),
                new SwitchSequential(Conv2d(640, 640, 3, stride: 2, padding: 1)),
                new SwitchSequential(new ResidualBlock(640, 1280), new AttentionBlock(8, 160)),
                new SwitchSequential(new ResidualBlock(1280, 1280), new AttentionBlock(8, 160)),
                new SwitchSequential(Conv2d(1280, 1280, 3, stride: 2, padding: 1)),
                new SwitchSequential(new ResidualBlock(1280, 1280)),
                new SwitchSequential(new ResidualBlock(1280, 1280))
                );
            bottleneck = new SwitchSequential(
                new ResidualBlock(1280, 1280),
                new AttentionBlock(8, 160),
                new ResidualBlock(1280, 1280)
                );
            decoders = new ModuleList<SwitchSequential>(
                new SwitchSequential(new ResidualBlock(2560, 1280)),
                new SwitchSequential(new ResidualBlock(2560, 1280)),
                new SwitchSequential(new ResidualBlock(2560, 1280), new Upsample(1280)),
                new SwitchSequential(new ResidualBlock(2560, 1280), new AttentionBlock(8, 160)),
                new SwitchSequential(new ResidualBlock(2560, 1280), new AttentionBlock(8, 160)),
                new SwitchSequential(new ResidualBlock(1920, 1280), new AttentionBlock(8, 160), new Upsample(1280)),
                new SwitchSequential(new ResidualBlock(1920, 640), new AttentionBlock(8, 80)),
                new SwitchSequential(new ResidualBlock(1280, 640), new AttentionBlock(8, 80)),
                new SwitchSequential(new ResidualBlock(960, 640), new AttentionBlock(8, 80), new Upsample(640)),
                new SwitchSequential(new ResidualBlock(960, 320), new AttentionBlock(8, 40)),
                new SwitchSequential(new ResidualBlock(640, 320), new AttentionBlock(8, 40)),
                new SwitchSequential(new ResidualBlock(640, 320), new AttentionBlock(8, 40))
                );
            RegisterComponents();
        }

        public override Tensor forward(Tensor x, Tensor context, Tensor time)
        {
            using var _ = NewDisposeScope();
            List<Tensor> skip_connections = new List<Tensor>();
            foreach (var layers in encoders)
            {
                x = layers.forward(x, context, time);
                skip_connections.Add(x);
            }
            x = bottleneck.forward(x, context, time);
            foreach (var layers in decoders)
            {
                var index = skip_connections.Last();
                x = torch.cat(new Tensor[] { x, index }, 1);
                skip_connections.RemoveAt(skip_connections.Count - 1);
                x = layers.forward(x, context, time);
            }
            return x.MoveToOuterDisposeScope();
        }
    }

    public class FinalLayer : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv;
        private readonly GroupNorm groupnorm;

        public FinalLayer(long in_channels, long out_channels) : base("FinalLayer")
        {
            groupnorm = GroupNorm(32, in_channels);
            conv = Conv2d(in_channels, out_channels, kernelSize: 3, padding: 1);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            x = groupnorm.forward(x);
            x = torch.nn.functional.silu(x);
            x = conv.forward(x);
            return x.MoveToOuterDisposeScope();
        }
    }

    public class Diffusion : Module<Tensor, Tensor, Tensor, Tensor>
    {
        private readonly TimeEmbedding time_embedding;
        private readonly UNet unet;
        private readonly FinalLayer final;

        public Diffusion() : base("Diffusion")
        {
            time_embedding = new TimeEmbedding(320);
            unet = new UNet();
            final = new FinalLayer(320, 4);
            RegisterComponents();
        }

        public override Tensor forward(Tensor latent, Tensor context, Tensor time)
        {
            using var _ = NewDisposeScope();
            time = time_embedding.forward(time);
            var output = unet.forward(latent, context, time);
            output = final.forward(output);
            return output.MoveToOuterDisposeScope();
        }
    }
}
