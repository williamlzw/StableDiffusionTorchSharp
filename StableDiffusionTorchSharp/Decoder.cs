using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionTorchSharp
{
	public class AttentionBlockA : Module<Tensor, Tensor>
	{
		internal readonly GroupNorm groupnorm;
		internal readonly SelfAttention attention;

		public AttentionBlockA(long channels) : base("AttentionBlockA")
		{
			groupnorm = GroupNorm(32, channels);
			attention = new SelfAttention(32, channels, causal_mask: true);
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
		internal readonly GroupNorm groupnorm_1;
		internal readonly Conv2d conv_1;
		internal readonly GroupNorm groupnorm_2;
		internal readonly Conv2d conv_2;
		internal readonly Module<Tensor, Tensor> residual_layer;
		internal readonly bool identity;

		public ResidualBlockA(long in_channels, long out_channels) : base("ResidualBlockA")
		{
			groupnorm_1 = GroupNorm(32, in_channels);
			conv_1 = Conv2d(in_channels, out_channels, kernelSize: 3, padding: 1);
			groupnorm_2 = GroupNorm(32, out_channels);
			conv_2 = Conv2d(out_channels, out_channels, kernelSize: 3, padding: 1);
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

			//mid
			new ResidualBlockA(512, 512),
			new AttentionBlockA(512),
			new ResidualBlockA(512, 512),

			// up
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
			GELU(),
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

			var t = tensors.First(a => a.Name == "first_stage_model.post_quant_conv.weight");
			this.to(t.Type);

			byte[] data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.post_quant_conv.weight"));
			((Conv2d)children().ToArray()[0]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.post_quant_conv.bias"));
			((Conv2d)children().ToArray()[0]).bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.conv_in.weight"));
			((Conv2d)children().ToArray()[1]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.conv_in.bias"));
			((Conv2d)children().ToArray()[1]).bias.bytes = data;

			// mid
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[2]), "first_stage_model.decoder.mid.block_1");
			ModelLoader.LoadData.LoadAttentionBlockA(modelLoader, tensors, (AttentionBlockA)(children().ToArray()[3]), "first_stage_model.decoder.mid");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[4]), "first_stage_model.decoder.mid.block_2");


			// first_stage_model.decoder.up.3
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[5]), "first_stage_model.decoder.up.3.block.0");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[6]), "first_stage_model.decoder.up.3.block.1");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[7]), "first_stage_model.decoder.up.3.block.2");

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.up.3.upsample.conv.weight"));
			((Conv2d)children().ToArray()[9]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.up.3.upsample.conv.bias"));
			((Conv2d)children().ToArray()[9]).bias.bytes = data;


			// first_stage_model.decoder.up.2
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[10]), "first_stage_model.decoder.up.2.block.0");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[11]), "first_stage_model.decoder.up.2.block.1");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[12]), "first_stage_model.decoder.up.2.block.2");

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.up.2.upsample.conv.weight"));
			((Conv2d)children().ToArray()[14]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.up.2.upsample.conv.bias"));
			((Conv2d)children().ToArray()[14]).bias.bytes = data;


			// first_stage_model.decoder.up.1
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[15]), "first_stage_model.decoder.up.1.block.0");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[16]), "first_stage_model.decoder.up.1.block.1");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[17]), "first_stage_model.decoder.up.1.block.2");

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.up.1.upsample.conv.weight"));
			((Conv2d)children().ToArray()[19]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.up.1.upsample.conv.bias"));
			((Conv2d)children().ToArray()[19]).bias.bytes = data;


			// first_stage_model.decoder.up.0
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[20]), "first_stage_model.decoder.up.0.block.0");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[21]), "first_stage_model.decoder.up.0.block.1");
			ModelLoader.LoadData.LoadResidualBlockA(modelLoader, tensors, (ResidualBlockA)(children().ToArray()[22]), "first_stage_model.decoder.up.0.block.2");



			// out
			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.norm_out.weight"));
			((GroupNorm)children().ToArray()[23]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.norm_out.bias"));
			((GroupNorm)children().ToArray()[23]).bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.conv_out.weight"));
			((Conv2d)children().ToArray()[25]).weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensors.First(a => a.Name == "first_stage_model.decoder.conv_out.bias"));
			((Conv2d)children().ToArray()[25]).bias.bytes = data;

			return this;
		}
	}
}
