using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace StableDiffusionTorchSharp
{
	public class TimeEmbedding : Module<Tensor, Tensor>
	{
		internal readonly Linear linear_1;
		internal readonly Linear linear_2;

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
		internal readonly GroupNorm groupnorm_feature;
		internal readonly Conv2d conv_feature;
		internal readonly Linear linear_time;
		internal readonly GroupNorm groupnorm_merged;
		internal readonly Conv2d conv_merged;
		internal readonly Module<Tensor, Tensor> residual_layer;
		internal readonly bool identity;

		public ResidualBlock(long in_channels, long out_channels, long n_time = 1280) : base("ResidualBlock")
		{
			groupnorm_feature = GroupNorm(32, in_channels);
			conv_feature = Conv2d(in_channels, out_channels, kernelSize: 3, padding: 1);
			linear_time = Linear(n_time, out_channels);
			groupnorm_merged = GroupNorm(32, out_channels);
			conv_merged = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
			identity = (in_channels == out_channels);
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
		internal readonly GroupNorm groupnorm;
		internal readonly Conv2d conv_input;
		internal readonly LayerNorm layernorm_1;
		internal readonly SelfAttention attention_1;
		internal readonly LayerNorm layernorm_2;
		internal readonly CrossAttention attention_2;
		internal readonly LayerNorm layernorm_3;
		internal readonly Linear linear_geglu_1;
		internal readonly Linear linear_geglu_2;
		internal readonly Conv2d conv_output;

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
		internal readonly Conv2d conv;
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
		internal readonly ModuleList<SwitchSequential> encoders;
		internal readonly SwitchSequential bottleneck;
		internal readonly ModuleList<SwitchSequential> decoders;

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
		internal readonly Conv2d conv;
		internal readonly GroupNorm groupnorm;

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
		internal readonly TimeEmbedding time_embedding;
		internal readonly UNet unet;
		internal readonly FinalLayer final;

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

			// Load Encoders
			var t = tensors.First(a => a.Name == "model.diffusion_model.time_embed.0.weight");
			this.to(t.Type);

			byte[] data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.time_embed.0.weight"));
			time_embedding.linear_1.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.time_embed.0.bias"));
			time_embedding.linear_1.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.time_embed.2.weight"));
			time_embedding.linear_2.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.time_embed.2.bias"));
			time_embedding.linear_2.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.0.0.weight"));
			((Conv2d)unet.encoders[0][0]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.0.0.bias"));
			((Conv2d)unet.encoders[0][0]).bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.3.0.op.weight"));
			((Conv2d)unet.encoders[3][0]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.3.0.op.bias"));
			((Conv2d)unet.encoders[3][0]).bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.6.0.op.weight"));
			((Conv2d)unet.encoders[6][0]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.6.0.op.bias"));
			((Conv2d)unet.encoders[6][0]).bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.9.0.op.weight"));
			((Conv2d)unet.encoders[9][0]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.input_blocks.9.0.op.bias"));
			((Conv2d)unet.encoders[9][0]).bias.bytes = data;


			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[1][0], "model.diffusion_model.input_blocks.1.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.encoders[1][1], "model.diffusion_model.input_blocks.1.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[2][0], "model.diffusion_model.input_blocks.2.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.encoders[2][1], "model.diffusion_model.input_blocks.2.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[4][0], "model.diffusion_model.input_blocks.4.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.encoders[4][1], "model.diffusion_model.input_blocks.4.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[5][0], "model.diffusion_model.input_blocks.5.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.encoders[5][1], "model.diffusion_model.input_blocks.5.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[7][0], "model.diffusion_model.input_blocks.7.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.encoders[7][1], "model.diffusion_model.input_blocks.7.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[8][0], "model.diffusion_model.input_blocks.8.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.encoders[8][1], "model.diffusion_model.input_blocks.8.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[10][0], "model.diffusion_model.input_blocks.10.0");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.encoders[11][0], "model.diffusion_model.input_blocks.11.0");

			// Load bottleneck

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.bottleneck[0], "model.diffusion_model.middle_block.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.bottleneck[1], "model.diffusion_model.middle_block.1");
			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.bottleneck[2], "model.diffusion_model.middle_block.2");


			// Load decoders 

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[0][0], "model.diffusion_model.output_blocks.0.0");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[1][0], "model.diffusion_model.output_blocks.1.0");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[2][0], "model.diffusion_model.output_blocks.2.0");
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.output_blocks.2.1.conv.weight"));
			((Upsample)unet.decoders[2][1]).conv.weight.bytes = data;
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.output_blocks.2.1.conv.bias"));
			((Upsample)unet.decoders[2][1]).conv.bias.bytes = data;

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[3][0], "model.diffusion_model.output_blocks.3.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[3][1], "model.diffusion_model.output_blocks.3.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[4][0], "model.diffusion_model.output_blocks.4.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[4][1], "model.diffusion_model.output_blocks.4.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[5][0], "model.diffusion_model.output_blocks.5.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[5][1], "model.diffusion_model.output_blocks.5.1");
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.output_blocks.5.2.conv.weight"));
			((Upsample)unet.decoders[5][2]).conv.weight.bytes = data;
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.output_blocks.5.2.conv.bias"));
			((Upsample)unet.decoders[5][2]).conv.bias.bytes = data;

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[6][0], "model.diffusion_model.output_blocks.6.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[6][1], "model.diffusion_model.output_blocks.6.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[7][0], "model.diffusion_model.output_blocks.7.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[7][1], "model.diffusion_model.output_blocks.7.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[8][0], "model.diffusion_model.output_blocks.8.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[8][1], "model.diffusion_model.output_blocks.8.1");
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.output_blocks.8.2.conv.weight"));
			((Upsample)unet.decoders[8][2]).conv.weight.bytes = data;
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.output_blocks.8.2.conv.bias"));
			((Upsample)unet.decoders[8][2]).conv.bias.bytes = data;

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[9][0], "model.diffusion_model.output_blocks.9.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[9][1], "model.diffusion_model.output_blocks.9.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[10][0], "model.diffusion_model.output_blocks.10.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[10][1], "model.diffusion_model.output_blocks.10.1");

			ModelLoader.LoadData.LoadResidualBlock(modelLoader, tensors, (ResidualBlock)unet.decoders[11][0], "model.diffusion_model.output_blocks.11.0");
			ModelLoader.LoadData.LoadAttentionBlock(modelLoader, tensors, (AttentionBlock)unet.decoders[11][1], "model.diffusion_model.output_blocks.11.1");

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.out.2.weight"));
			final.conv.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.out.2.bias"));
			final.conv.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.out.0.weight"));
			final.groupnorm.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "model.diffusion_model.out.0.bias"));
			final.groupnorm.bias.bytes = data;


			return this;
		}

	}



}
