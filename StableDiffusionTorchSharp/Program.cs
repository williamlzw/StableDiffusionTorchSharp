using TorchSharp;
using static TorchSharp.torch;

namespace StableDiffusionTorchSharp
{
    public class Program
    {
        public static void Main()
        {
            Generate();
        }

        public static Tensor GetTimeEnmedding(float timestep)
        {
            var freqs = torch.pow(10000, -torch.arange(0, 160, dtype: torch.float32) / 160);
            var x = torch.tensor(new float[] { timestep }, dtype: torch.float32)[torch.TensorIndex.Colon, torch.TensorIndex.None] * freqs[torch.TensorIndex.None];
            return torch.cat(new Tensor[] { torch.cos(x), torch.sin(x) }, dim: -1);
        }

        public static void Generate()
        {
            using (torch.no_grad())
            {
                int num_inference_steps = 15;
                var device = torch.device("cpu");
                torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
                var clip = new CLIP();
                clip.load("model\\clip.dat");
                clip.eval();

                string VocabPath = "model\\vocab.json";
                string MergesPath = "model\\merges.txt";
                var tokenizer = new Tokenizer(VocabPath, MergesPath);

                string prompt = "typographic art bird. stylized, intricate, detailed, artistic, text-based";
                string uncond_prompts = "";

                var cond_tokens_ids = tokenizer.Encode(prompt);
                var cond_tokens = torch.tensor(cond_tokens_ids, torch.@long).unsqueeze(0);
                var cond_context = clip.forward(cond_tokens);

                var uncond_tokens_ids = tokenizer.Encode(uncond_prompts);
                var uncond_tokens = torch.tensor(uncond_tokens_ids, torch.@long).unsqueeze(0);
                var uncond_context = clip.forward(uncond_tokens);

                var context = torch.cat(new Tensor[] { cond_context, uncond_context }).to(device);

                torch.save(context, "context.dat");
                //var context = torch.load("context.dat");

                var diffusion = new Diffusion();
                diffusion.load("model\\unet.dat").to(device);
                diffusion.eval();
                long[] noise_shape = new long[] { 1, 4, 64, 64 };
                var latents = torch.randn(noise_shape, device: device);
                var sampler = new EulerDiscreteScheduler();

                sampler.SetTimesteps(num_inference_steps, device);
                latents *= sampler.InitNoiseSigma();

                for (int i = 0; i < num_inference_steps; i++)
                {
                    Console.WriteLine($"begin step");
                    var timestep = sampler.timesteps_[i];
                    var time_embedding = GetTimeEnmedding(timestep.ToSingle()).to(device);
                    var input_latents = sampler.ScaleModelInput(latents, timestep);
                    input_latents = input_latents.repeat(2, 1, 1, 1).to(device);
                    var output = diffusion.forward(input_latents, context, time_embedding);
                    var ret = output.chunk(2);
                    var output_cond = ret[0];
                    var output_uncond = ret[1];
                    output = 7.5 * (output_cond - output_uncond) + output_uncond;
                    latents = sampler.Step(output, timestep, latents);
                    torch.save(latents, $"latent{i}.dat");
                    Console.WriteLine($"end step, {i}");
                }

                //var latents = torch.load("latent14.dat");
                var decoder = new Decoder();
                decoder.load("model\\decoder.dat");
                decoder.eval();
                Console.WriteLine($"begin decoder");
                var images = decoder.forward(latents.cpu());
                Console.WriteLine($"end decoder");
                images = images.clip(-1, 1) * 0.5 + 0.5;
                images = torchvision.transforms.functional.convert_image_dtype(images, torch.ScalarType.Byte);
                torchvision.io.write_jpeg(images, "result1.jpg");
            }
        }
    }
}