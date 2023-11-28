using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace StableDiffusionTorchSharp
{
    public class CLIPEmbedding : Module<Tensor, Tensor>
    {
        private readonly Embedding token_embedding;
        private readonly Parameter position_value;
        public CLIPEmbedding(long n_vocab, long n_embd, long n_token) : base("CLIPEmbedding")
        {
            token_embedding = Embedding(n_vocab, n_embd);
            position_value = Parameter(torch.zeros(n_token, n_embd));
            RegisterComponents();
        }

        public override Tensor forward(Tensor tokens)
        {
            using var _ = NewDisposeScope();
            var x = token_embedding.forward(tokens);
            x += position_value;
            return x.MoveToOuterDisposeScope();
        }
    }

    public class CLIPLayer : Module<Tensor, Tensor>
    {
        private readonly LayerNorm layernorm_1;
        private readonly LayerNorm layernorm_2;
        private readonly SelfAttention attention;
        private readonly Linear linear_1;
        private readonly Linear linear_2;

        public CLIPLayer(long n_head, long n_embd) : base("CLIPLayer")
        {
            layernorm_1 = LayerNorm(n_embd);
            attention = new SelfAttention(n_head, n_embd, causal_mask: true);
            layernorm_2 = LayerNorm(n_embd);
            linear_1 = Linear(n_embd, 4 * n_embd);
            linear_2 = Linear(4 * n_embd, n_embd);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var _ = NewDisposeScope();
            var residue = x;
            x = layernorm_1.forward(x);
            x = attention.forward(x);
            x += residue;
            residue = x;
            x = layernorm_2.forward(x);
            x = linear_1.forward(x);
            x = x * torch.sigmoid(1.702 * x);
            x = linear_2.forward(x);
            x += residue;
            return x.MoveToOuterDisposeScope();
        }
    }

    public class CLIP : Module<Tensor, Tensor>
    {
        private readonly CLIPEmbedding embedding;
        private readonly ModuleList<Module<Tensor, Tensor>> layers;
        private readonly LayerNorm layernorm;

        public CLIP() : base("CLIP")
        {
            embedding = new CLIPEmbedding(49408, 768, 77);
            layers = nn.ModuleList<Module<Tensor, Tensor>>();
            for (int i = 0; i < 12; i++)
            {
                layers.Add(new CLIPLayer(12, 768));
            }
            layernorm = LayerNorm(768);
            RegisterComponents();
        }

        public override Tensor forward(Tensor token)
        {
            using var _ = NewDisposeScope();
            var state = embedding.forward(token);
            foreach (var layer in layers)
            {
                state = layer.forward(state);
            }
            var output = layernorm.forward(state);
            return output.MoveToOuterDisposeScope();
        }
    }
}
