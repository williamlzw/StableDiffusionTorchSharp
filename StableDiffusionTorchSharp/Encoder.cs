using System.Collections.Immutable;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace StableDiffusionTorchSharp
{
    public class Tokenizer
    {
        private IReadOnlyDictionary<string, int> _encoder;
        private IReadOnlyDictionary<int, string> _decoder;
        private IReadOnlyDictionary<int, string> _byteEncoder;
        private IReadOnlyDictionary<string, int> _byteDecoder;
        private IReadOnlyDictionary<(string, string), int> _bpeRanks;
        private int _bosToken;
        private int _eosToken;
        private int _padToken;
        private int _maxLength;
        private Regex _chunk_pattern;

        public Tokenizer(string vocabPath, string mergesPath)
        {
            var str = File.ReadAllText(vocabPath);
            _encoder = (Dictionary<string, int>)JsonSerializer.Deserialize(str, typeof(Dictionary<string, int>));
            _byteEncoder = bytesToUnicode();
            _decoder = _encoder.ToDictionary(x => (int)x.Value, x => x.Key);
            _byteDecoder = _byteEncoder.ToDictionary(x => x.Value, x => (int)x.Key);
            _bosToken = _encoder["<|startoftext|>"];
            _eosToken = _encoder["<|endoftext|>"];
            _padToken = _encoder["<|endoftext|>"];
            _maxLength = 77;

            Dictionary<(string, string), int> ranks = new();
            var bpeStr = File.ReadAllText(mergesPath);
            var lines = bpeStr.Trim().Split("\n", StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).Skip(1);
            var i = 0;

            foreach (string s in lines)
            {
                var t = s.Split(" ").ToArray();
                ranks[(t[0], t[1])] = i++;
            }
            _bpeRanks = ranks;
            string pattern = @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";
            _chunk_pattern = new Regex(pattern, RegexOptions.IgnoreCase);
        }

        public List<int> Encode(string text)
        {
            text = Regex.Replace(text, @"\\s+", " ");
            text = text.Trim();
            text = text.ToLower();

            List<int> bpeTokens = new List<int>();
            bpeTokens.Add(_bosToken);


            foreach (var token in _chunk_pattern.Matches(text).Select(m => m.Value))
            {
                var tmp = string.Join("", Encoding.UTF8.GetBytes(token).Select(b => _byteEncoder[b]));
                var newTokens = bpe(tmp).Select(x => _encoder[x]);
                bpeTokens.AddRange(newTokens);
            }

            bpeTokens.Add(_eosToken);

            if (bpeTokens.Count > _maxLength)
            {
                bpeTokens = bpeTokens.GetRange(0, _maxLength);
            }

            if (bpeTokens.Count < _maxLength)
            {
                bpeTokens.AddRange(Enumerable.Repeat(_padToken, _maxLength - bpeTokens.Count));
            }

            return bpeTokens;
        }

        public string Decode(List<int> tokens)
        {
            return Encoding.UTF8.GetString(string.Join("", tokens.Select(x => _decoder[x])).ToCharArray().Select(x => (byte)_byteDecoder[x.ToString()]).ToArray()).Replace("<|startoftext|>", "").Replace("<|endoftext|>", "").Replace("</w>", " ");
        }

        private IEnumerable<(string, string)> GetPairs(List<string> word)
        {
            var prev = word.First();
            foreach (var s in word.Skip(1))
            {
                yield return (prev, s);
                prev = s;
            }
        }

        private Dictionary<int, string> bytesToUnicode()
        {
            var bs = Enumerable.Range((int)'!', (int)'~' - (int)'!' + 1).
                Concat(Enumerable.Range((int)'¡', (int)'¬' - (int)'¡' + 1)).
                Concat(Enumerable.Range((int)'®', (int)'ÿ' - (int)'®' + 1));

            var cs = bs.Select(x => x);
            var n = 0;
            for (var b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs = bs.Append(b);
                    cs = cs.Append(256 + n);
                    n = n + 1;
                }
            }

            var tmp = cs.Select(x => ((char)x).ToString());

            Dictionary<int, string> result = new Dictionary<int, string>();
            for (int i = 0; i < bs.Count(); i++)
                result[bs.ElementAt(i)] = tmp.ElementAt(i);
            return result;
        }

        private List<string> bpe(string token)
        {
            var words = token.ToCharArray().Select(c => c.ToString()).ToList();
            words[^1] += "</w>";
            while (words.Count > 1)
            {
                var pairs = GetPairs(words);
                var minPairs = new Dictionary<int, (string, string)>();
                int rank;
                pairs.ToList().ForEach(p => minPairs[_bpeRanks.TryGetValue(p, out rank) ? rank : int.MaxValue] = p);

                var bigram = minPairs[minPairs.Min(p => p.Key)];

                if (!_bpeRanks.ContainsKey(bigram))
                {
                    break;
                }

                string first = bigram.Item1;
                string second = bigram.Item2;

                List<string> newWords = new List<string>();

                foreach (string word in words)
                {
                    if (word == second && newWords.Count > 0 && newWords[^1] == first)
                    {
                        newWords[^1] = first + second;
                    }
                    else
                    {
                        newWords.Add(word);
                    }
                }

                words = newWords;

            }
            return words;
        }
    }
}