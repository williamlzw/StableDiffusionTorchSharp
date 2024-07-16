using TorchSharp.Modules;

namespace StableDiffusionTorchSharp.ModelLoader
{
	internal class LoadData
	{
		public static void LoadResidualBlock(IModelLoader modelLoader, List<Tensor> tensorlist, ResidualBlock modules, string name, bool Identity = true)
		{
			byte[] data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".in_layers.0.weight"));
			modules.groupnorm_feature.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".in_layers.0.bias"));
			modules.groupnorm_feature.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".in_layers.2.weight"));
			modules.conv_feature.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".in_layers.2.bias"));
			modules.conv_feature.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".emb_layers.1.weight"));
			modules.linear_time.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".emb_layers.1.bias"));
			modules.linear_time.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".out_layers.0.weight"));
			modules.groupnorm_merged.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".out_layers.0.bias"));
			modules.groupnorm_merged.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".out_layers.3.weight"));
			modules.conv_merged.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".out_layers.3.bias"));
			modules.conv_merged.bias.bytes = data;

			if (!Identity)
			{
				data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".skip_connection.weight"));
				((Conv2d)modules.residual_layer).weight.bytes = data;

				data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".skip_connection.bias"));
				((Conv2d)modules.residual_layer).bias.bytes = data;
			}

		}

		public static void LoadAttentionBlock(IModelLoader modelLoader, List<Tensor> tensorlist, AttentionBlock modules, string name)
		{
			byte[] data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".norm.weight"));
			modules.groupnorm.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".norm.bias"));
			modules.groupnorm.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".proj_in.weight"));
			modules.conv_input.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".proj_in.bias"));
			modules.conv_input.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.norm1.weight"));
			modules.layernorm_1.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.norm1.bias"));
			modules.layernorm_1.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.norm2.weight"));
			modules.layernorm_2.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.norm2.bias"));
			modules.layernorm_2.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.norm3.weight"));
			modules.layernorm_3.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.norm3.bias"));
			modules.layernorm_3.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn1.to_q.weight"));
			modules.attention_1.q_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn1.to_k.weight"));
			modules.attention_1.k_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn1.to_v.weight"));
			modules.attention_1.v_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn1.to_out.0.weight"));
			modules.attention_1.out_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn1.to_out.0.bias"));
			modules.attention_1.out_proj.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn2.to_q.weight"));
			modules.attention_2.q_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn2.to_k.weight"));
			modules.attention_2.k_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn2.to_v.weight"));
			modules.attention_2.v_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn2.to_out.0.weight"));
			modules.attention_2.out_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.attn2.to_out.0.bias"));
			modules.attention_2.out_proj.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.ff.net.0.proj.weight"));
			modules.linear_geglu_1.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.ff.net.0.proj.bias"));
			modules.linear_geglu_1.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.ff.net.2.weight"));
			modules.linear_geglu_2.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".transformer_blocks.0.ff.net.2.bias"));
			modules.linear_geglu_2.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".proj_out.weight"));
			modules.conv_output.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".proj_out.bias"));
			modules.conv_output.bias.bytes = data;

		}


		public static void LoadResidualBlockA(IModelLoader modelLoader, List<Tensor> tensorlist, ResidualBlockA modules, string name, bool Identity = true)
		{
			byte[] data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".conv1.weight"));
			modules.conv_1.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".conv1.bias"));
			modules.conv_1.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".conv2.weight"));
			modules.conv_2.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".conv2.bias"));
			modules.conv_2.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".norm1.weight"));
			modules.groupnorm_1.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".norm1.bias"));
			modules.groupnorm_1.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".norm2.weight"));
			modules.groupnorm_2.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".norm2.bias"));
			modules.groupnorm_2.bias.bytes = data;

			if (!Identity)
			{
				data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".nin_shortcut.weight"));
				((Conv2d)modules.residual_layer).weight.bytes = data;

				data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".nin_shortcut.bias"));
				((Conv2d)modules.residual_layer).bias.bytes = data;
			}

		}

		public static void LoadAttentionBlockA(IModelLoader modelLoader, List<Tensor> tensorlist, AttentionBlockA modules, string name)
		{
			byte[] data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.norm.weight"));
			modules.groupnorm.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.norm.bias"));
			modules.groupnorm.bias.bytes = data;


			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.q.weight"));
			modules.attention.q_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.q.bias"));
			modules.attention.q_proj.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.k.weight"));
			modules.attention.k_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.k.bias"));
			modules.attention.k_proj.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.v.weight"));
			modules.attention.v_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.v.bias"));
			modules.attention.v_proj.bias.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.proj_out.weight"));
			modules.attention.out_proj.weight.bytes = data;

			data = modelLoader.ReadByteFromFile(tensorlist.First(a => a.Name == name + ".attn_1.proj_out.bias"));
			modules.attention.out_proj.bias.bytes = data;

		}

	}
}
