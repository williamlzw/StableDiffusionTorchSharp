using Microsoft.ML.Tokenizers;

namespace StableDiffusionTorchSharp
{
    public class ClipTokenizer
    {
        private readonly Tokenizer _tokenizer;
        private readonly int _startToken;
        private readonly int _endToken;

        public ClipTokenizer(string vocabPath, string mergesPath, int startToken = 49406, int endToken = 49407)
        {
            _tokenizer = new Tokenizer(new Bpe(vocabPath, mergesPath, endOfWordSuffix: "</w>"));
            _startToken = startToken;
            _endToken = endToken;
        }

        public List<int> Tokenize(string text, int maxTokens = 77)
        {
            var res = _tokenizer.Encode(text);
            var tokens = new[] { _startToken }.Concat(res.Ids.Concat(Enumerable.Repeat(0, maxTokens - res.Ids.Count - 2))).Concat(new[] { _endToken }).ToArray();
            return new List<int>(tokens);
        }
    }
}