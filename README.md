# MicroASM
Aspect sentiment model for short text

This code was used in the experiments of the research paper

**Reinald Kim Amplayo** and Seung-won Hwang. **Aspect Sentiment Model for Micro Reviews**. _ICDM_, 2017.

To run the code, several parameters are needed to be set. Refer to our paper to determine the recommended values:
- `file`: The file containing the short text, one per line.
- `seedDir`: The directory containing the sentiment seed words, containing two (or more if you have more sentiments, in the paper we only use positive and negative as sentiment labels) files named `0.txt` and `1.txt`. These files should contain one sentiment lexicon per line. For a sample seed word lists, refer to the `data` folder.
- `noOfTopics`: The number of topics/aspects.
- `noOfSentiments`: The number of sentiments.
- `noOfPseudodocs`: The number of clusters.
- `noOfIters`: The number of iterations.
- `alpha`: The Dirichlet prior on the aspect-sentiment distribution.
- `beta0`, `beta1`, `beta2`: The different Dirichlet priors on the word distribution that depends on the sentiment seed words given.
- `gamma`: The Dirichlet prior on the sentiment distribution.
- `delta`: The Dirichlet prior on the document distribution.
- `window`: The context window to create aspect-sentiment pairs.

An example run would be:

`java MicroASM data.txt seed/ 15 2 500 1500 0.1 0.01 0.1 0 1 0.1 5`

To cite the paper/code, please use this BibTex:

```
@inproceedings{amplayo2017aspect,
	Author = {Reinald Kim Amplayo and Seung-won Hwang},
	Booktitle = {ICDM},
	Location = {Osaka, Japan},
	Year = {2016},
	Title = {Building Content-driven Entity Networks for Scarce Scientific Literature using Content Information},
}
```

If you have questions, send me an email: rktamplayo at yonsei dot ac dot kr
