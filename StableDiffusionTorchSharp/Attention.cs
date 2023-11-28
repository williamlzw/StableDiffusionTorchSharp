using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace StableDiffusionTorchSharp
{
    public class SelfAttention : Module<Tensor, Tensor>
    {
        private readonly Linear in_proj;
        private readonly Linear out_proj;
        private readonly long n_heads_;
        private readonly long d_head;
        private readonly bool causal_mask_;

        public SelfAttention(long n_heads, long d_embed, bool in_proj_bias = true, bool out_proj_bias = true, bool causal_mask = false) : base("SelfAttention")
        {
            in_proj = Linear(d_embed, 3 * d_embed, hasBias: in_proj_bias);
            out_proj = Linear(d_embed, d_embed, hasBias: out_proj_bias);
            n_heads_ = n_heads;
            d_head = d_embed / n_heads;
            causal_mask_ = causal_mask;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            var input_shape = x.shape;
            var batch_size = input_shape[0];
            var sequence_length = input_shape[1];
            var ret = in_proj.forward(x).chunk(3, dim: -1);
            long[] interim_shape = new long[] { batch_size, sequence_length, n_heads_, d_head };
            var q = ret[0];
            var k = ret[1];
            var v = ret[2];
            q = q.view(interim_shape).transpose(1, 2);
            k = k.view(interim_shape).transpose(1, 2);
            v = v.view(interim_shape).transpose(1, 2);
            var weight = torch.matmul(q, k.transpose(-1, -2));
            if (causal_mask_)
            {
                var mask = torch.ones_like(weight).triu(1).to(torch.@bool);
                weight.masked_fill_(mask, System.Single.NegativeInfinity);
            }
            weight = weight / (float)Math.Sqrt(d_head);
            weight = torch.nn.functional.softmax(weight, dim: -1);

            var output = torch.matmul(weight, v);
            output = output.transpose(1, 2);
            output = output.reshape(input_shape);
            output = out_proj.forward(output);
            return output.MoveToOuterDisposeScope();
        }
    }

    public class CrossAttention : Module<Tensor, Tensor, Tensor>
    {
        private readonly Linear q_proj;
        private readonly Linear k_proj;
        private readonly Linear v_proj;
        private readonly Linear out_proj;
        private readonly long n_heads_;
        private readonly long d_head;

        public CrossAttention(long n_heads, long d_embed, long d_cross, bool in_proj_bias = true, bool out_proj_bias = true) : base("CrossAttention")
        {
            q_proj = Linear(d_embed, d_embed, hasBias: in_proj_bias);
            k_proj = Linear(d_cross, d_embed, hasBias: in_proj_bias);
            v_proj = Linear(d_cross, d_embed, hasBias: in_proj_bias);
            out_proj = Linear(d_embed, d_embed, hasBias: out_proj_bias);
            n_heads_ = n_heads;
            d_head = d_embed / n_heads;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x, Tensor y)
        {
            using var _ = NewDisposeScope();
            var input_shape = x.shape;
            var batch_size = input_shape[0];
            var sequence_length = input_shape[1];

            long[] interim_shape = new long[] { batch_size, -1, n_heads_, d_head };
            var q = q_proj.forward(x);
            var k = k_proj.forward(y);
            var v = v_proj.forward(y);

            q = q.view(interim_shape).transpose(1, 2);
            k = k.view(interim_shape).transpose(1, 2);
            v = v.view(interim_shape).transpose(1, 2);
            var weight = torch.matmul(q, k.transpose(-1, -2));
            weight = weight / (float)Math.Sqrt(d_head);
            weight = torch.nn.functional.softmax(weight, dim: -1);

            var output = torch.matmul(weight, v);
            output = output.transpose(1, 2).contiguous();
            output = output.reshape(input_shape);
            output = out_proj.forward(output);
            return output.MoveToOuterDisposeScope();
        }
    }
}
