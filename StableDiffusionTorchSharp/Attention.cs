using TorchSharp;
using TorchSharp.FlashAttention;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace StableDiffusionTorchSharp
{
	public class SelfAttention : Module<Tensor, Tensor>
	{
		internal readonly Linear in_proj;
		internal readonly Linear out_proj;
		private readonly long n_heads_;
		private readonly long d_head;
		bool causal_mask_;
		float dropout_p;
		bool useFlashAtten;
		Dropout dropout;

		public SelfAttention(long n_heads, long d_embed, bool in_proj_bias = true, bool out_proj_bias = true, bool causal_mask = false, float dropout_p = 0.1f, bool useFlashAtten = true) : base("SelfAttention")
		{
			in_proj = Linear(d_embed, 3 * d_embed, hasBias: in_proj_bias);
			out_proj = Linear(d_embed, d_embed, hasBias: out_proj_bias);
			n_heads_ = n_heads;
			d_head = d_embed / n_heads;
			causal_mask_ = causal_mask;
			this.dropout_p = dropout_p;
			this.useFlashAtten = useFlashAtten;
			dropout = Dropout(dropout_p);
			RegisterComponents();
		}

		public override Tensor forward(Tensor x)
		{
			if (useFlashAtten)
			{
				using var _ = NewDisposeScope();
				var input_shape = x.shape;
				var batch_size = input_shape[0];
				var sequence_length = input_shape[1];
				long[] interim_shape = new long[] { batch_size, sequence_length, 3, n_heads_, d_head };
				var output = in_proj.forward(x);
				output = output.view(interim_shape);
				output = new FlashAttention(softmax_scale: 1 / (float)Math.Sqrt(d_head), dropout_p, causal_mask_).forward(output);
				output = output.reshape(input_shape);
				output = out_proj.forward(output);
				return output.MoveToOuterDisposeScope();
			}
			else
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
					weight.masked_fill_(mask, Single.NegativeInfinity);
				}
				weight = weight / (float)Math.Sqrt(d_head);
				weight = torch.nn.functional.softmax(weight, dim: -1);
				weight = dropout.forward(weight);

				var output = torch.matmul(weight, v);
				output = output.transpose(1, 2);
				output = output.reshape(input_shape);
				output = out_proj.forward(output);
				return output.MoveToOuterDisposeScope();
			}
		}
	}

	public class CrossAttention : Module<Tensor, Tensor, Tensor>
	{
		internal readonly Linear q_proj;
		internal readonly Linear k_proj;
		internal readonly Linear v_proj;
		internal readonly Linear out_proj;
		internal readonly long n_heads_;
		internal readonly long d_head;
		bool causal_mask_;
		bool useFlashAtten;
		float dropout_p;

		public CrossAttention(long n_heads, long d_embed, long d_cross, bool in_proj_bias = true, bool out_proj_bias = true, float dropout_p = 0.2f, bool causal_mask = true, bool useFlashAtten = false) : base("CrossAttention")
		{
			q_proj = Linear(d_embed, d_embed, hasBias: in_proj_bias);
			k_proj = Linear(d_cross, d_embed, hasBias: in_proj_bias);
			v_proj = Linear(d_cross, d_embed, hasBias: in_proj_bias);
			out_proj = Linear(d_embed, d_embed, hasBias: out_proj_bias);
			n_heads_ = n_heads;
			d_head = d_embed / n_heads;
			causal_mask_ = causal_mask;
			this.useFlashAtten = useFlashAtten;
			this.dropout_p = dropout_p;
			RegisterComponents();
		}

		public override Tensor forward(Tensor x, Tensor y)
		{
			if (useFlashAtten)
			{
				using var _ = NewDisposeScope();
				var input_shape = x.shape;
				var batch_size = input_shape[0];
				var sequence_length = input_shape[1];

				long[] interim_shape = new long[] { batch_size, -1, n_heads_, d_head };
				var q = q_proj.forward(x);
				var k = k_proj.forward(y);
				var v = v_proj.forward(y);

				q = q.view(interim_shape);
				k = k.view(interim_shape);
				v = v.view(interim_shape);

				(var output, var _, var _) = FlashAttentionInterface.flash_attn_func(q, k, v, dropout_p: dropout_p, softmax_scale: 1 / (float)Math.Sqrt(d_head), causal: causal_mask_);

				output = output.reshape(input_shape);
				output = out_proj.forward(output);
				return output.MoveToOuterDisposeScope();
			}
			else
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

				weight = Dropout(dropout_p).forward(weight);

				var output = torch.matmul(weight, v);
				output = output.transpose(1, 2).contiguous();
				output = output.reshape(input_shape);
				output = out_proj.forward(output);
				return output.MoveToOuterDisposeScope();
			}

		}

	}
}
