using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionTorchSharp
{
	internal class CLIPEmbedding : Module<Tensor, Tensor>
	{
		internal readonly Embedding token_embedding;
		internal readonly Parameter position_value;
		internal CLIPEmbedding(long n_vocab, long n_embd, long n_token) : base("CLIPEmbedding")
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

	internal class CLIPLayer : Module<Tensor, Tensor>
	{
		internal readonly LayerNorm layernorm_1;
		internal readonly LayerNorm layernorm_2;
		internal readonly SelfAttention attention;
		internal readonly Linear linear_1;
		internal readonly Linear linear_2;

		internal CLIPLayer(long n_head, long n_embd) : base("CLIPLayer")
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

	internal class CLIP : Module<Tensor, Tensor>
	{
		internal readonly CLIPEmbedding embedding;
		internal readonly ModuleList<Module<Tensor, Tensor>> layers;
		internal readonly LayerNorm layernorm;

		internal CLIP() : base("CLIP")
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

		public override Module load(string filename, bool strict = true, IList<string> skip = null, Dictionary<string, bool> loadedParameters = null)
		{
			string extension = Path.GetExtension(filename);

			if (extension.Contains("dat"))
			{
				using (FileStream fileStream = new FileStream(filename, FileMode.Open))
				{
					using (BinaryReader binaryReader = new BinaryReader(fileStream))
					{
						return load(binaryReader, strict, skip, loadedParameters);
					}
				}
			}

			ModelLoader.IModelLoader modelLoader = null;
			if (extension.Contains("safetensor"))
			{
				modelLoader = new ModelLoader.SafetensorsLoader();
			}
			else if (extension.Contains("pt"))
			{
				modelLoader = new ModelLoader.PickleLoader();
			}

			List<ModelLoader.Tensor> tensors = modelLoader.ReadTensorsInfoFromFile(filename);

			byte[] data;
			var t = tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight");
			this.to(t.Type);
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"));
			embedding.token_embedding.weight.bytes = new Span<byte>(data);

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"));
			embedding.position_value.bytes = new Span<byte>(data);

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.final_layer_norm.bias"));
			layernorm.bias.bytes = new Span<byte>(data);

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.final_layer_norm.weight"));
			layernorm.weight.bytes = new Span<byte>(data);

			for (int i = 0; i < layers.Count; i++)
			{
				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".layer_norm1.weight"));
				((CLIPLayer)layers[i]).layernorm_1.weight.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".layer_norm1.bias"));
				((CLIPLayer)layers[i]).layernorm_1.bias.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".layer_norm2.weight"));
				((CLIPLayer)layers[i]).layernorm_2.weight.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".layer_norm2.bias"));
				((CLIPLayer)layers[i]).layernorm_2.bias.bytes = new Span<byte>(data);


				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".mlp.fc1.weight"));
				((CLIPLayer)layers[i]).linear_1.weight.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".mlp.fc1.bias"));
				((CLIPLayer)layers[i]).linear_1.bias.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".mlp.fc2.weight"));
				((CLIPLayer)layers[i]).linear_2.weight.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".mlp.fc2.bias"));
				((CLIPLayer)layers[i]).linear_2.bias.bytes = new Span<byte>(data);


				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.q_proj.weight")).Concat(modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.k_proj.weight"))).Concat(modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.v_proj.weight"))).ToArray();
				((CLIPLayer)layers[i]).attention.in_proj.weight.bytes = data;

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.q_proj.bias")).Concat(modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.k_proj.bias"))).Concat(modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.v_proj.bias"))).ToArray();
				((CLIPLayer)layers[i]).attention.in_proj.bias.bytes = data;


				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.out_proj.weight"));
				((CLIPLayer)layers[i]).attention.out_proj.weight.bytes = new Span<byte>(data);

				data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "cond_stage_model.transformer.text_model.encoder.layers." + i + ".self_attn.out_proj.bias"));
				((CLIPLayer)layers[i]).attention.out_proj.bias.bytes = new Span<byte>(data);
			}

			return this;
		}

	}
}
