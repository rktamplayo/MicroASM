package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;

/**
 * Micro Aspect Sentiment Model (MicroASM)
 * @author Reinald Kim Amplayo
 * 
 * File input:
 * One line contains one document
 * One document contains list of words, separated by space
 * 
 * Seed directory:
 * Inside the directory, list of files named 0.txt, 1.txt, ..., (noOfSentiments-1).txt
 * Each line in the file represents a seed word for that specific sentiment
 */
public class MicroASM {

	// parameters
	static int noOfDocs, noOfWords, noOfIters, noOfTopics, noOfSentiments, noOfPseudodocs, noOfBiterms;
	static int window;
	static double alpha, gamma, delta, epsilon;
	static double[] betas;
	static double[] betaSums;
	
	// data
	static Vector<String> documents;
	static Vector<Double> scores;
	static TreeMap<String, Integer> wordToIdxMap;
	static TreeMap<Integer, String> idxToWordMap;
	static TreeMap<String, Integer> bitermToIdxMap;
	static TreeMap<Integer, String> idxToBitermMap;
	
	// seeds
	static Vector<TreeSet<Integer>> seeds;
	static TreeSet<Integer> allSeeds;
	
	// counts
	static int[] pseudodocCount;
	static int[][][] pseudodocSentimentTopicCount;
	static int[][] pseudodocSentimentCount;
	static int[][][] docSentimentTopicCount;
	static int[][] docSentimentCount;
	static int[][][] sentimentTopicWordCount;
	
	static int sumPseudodocCount;
	static int[][] sumPseudodocSentimentTopicCount;
	static int[] sumPseudodocSentimentCount;
	static int[][] sumDocSentimentTopicCount;
	static int[] sumDocSentimentCount;
	static int[][] sumSentimentTopicWordCount;
	
	// variables
	static Vector<TreeMap<Integer, TreeMap<Integer, Integer>>> topicMaps;
	static Vector<TreeMap<Integer, Integer>> sentimentMaps;
	static Vector<Vector<Integer>> topics, sentiments;
	static Vector<Vector<Integer>> words;
	static Vector<Integer> pseudodocs;
	
	// probability
	static double[][] multimatrix;
	static double[] multinomial;
	
	static String filename;
	
	public static void main(String[] args) throws Exception {
		String file = args[0];
		String seedDir = args[1];
		noOfTopics = Integer.parseInt(args[2]);
		noOfSentiments = Integer.parseInt(args[3]);
		noOfPseudodocs = Integer.parseInt(args[4]);
		noOfIters = Integer.parseInt(args[5]);
		alpha = Double.parseDouble(args[6]);
		betas = new double[3];
		betas[0] = Double.parseDouble(args[7]);
		betas[1] = Double.parseDouble(args[8]);
		betas[2] = Double.parseDouble(args[9]);
		gamma = Double.parseDouble(args[10]);
		delta = Double.parseDouble(args[11]);
		window = Integer.parseInt(args[12]);
		
		seeds = new Vector<TreeSet<Integer>>();
		allSeeds = new TreeSet<Integer>();
		wordToIdxMap = new TreeMap<String, Integer>();
		idxToWordMap = new TreeMap<Integer, String>();
		bitermToIdxMap = new TreeMap<String, Integer>();
		idxToBitermMap = new TreeMap<Integer, String>();
		
		noOfWords = 0;
		noOfBiterms = 0;
		
		for(int i=0; i<noOfSentiments; i++) {
			TreeSet<Integer> seed = new TreeSet<Integer>();
			BufferedReader in = new BufferedReader(new FileReader(seedDir + "/" + i + ".txt"));
			while(in.ready()) {
				String word = in.readLine();
				if(!wordToIdxMap.containsKey(word)) {
					wordToIdxMap.put(word, noOfWords);
					idxToWordMap.put(noOfWords, word);
					noOfWords++;
				}
				int widx = wordToIdxMap.get(word);
				seed.add(widx);
				allSeeds.add(widx);
			}
			in.close();
			seeds.add(seed);
		}
		
		BufferedReader in = new BufferedReader(new FileReader(file));
		documents = new Vector<String>();
		scores = new Vector<Double>();
		words = new Vector<Vector<Integer>>();
		
		noOfDocs = 0;
		
		while(in.ready()) {
			String[] line = in.readLine().split("\t");
			
			double score = Double.parseDouble(line[1]);
			scores.add(score);
			
			String str = line[0];
			String[] spl = str.split(" ");
			
			Vector<Integer> wordList = new Vector<Integer>();
			for(int i=0; i<spl.length; i++)
				spl[i] = spl[i].substring(0, spl[i].length()-2);
			for(int i=0; i<spl.length; i++) {
				String word = spl[i];
				if(!wordToIdxMap.containsKey(word)) {
					wordToIdxMap.put(word, noOfWords);
					idxToWordMap.put(noOfWords, word);
					noOfWords++;
				}
				int widx = wordToIdxMap.get(word);
				wordList.add(widx);
				for(int j=Math.max(0, i-window); j<=Math.min(spl.length-1, i+window); j++) {
					String word2 = spl[j];
					String word1 = word;
					if(!bitermToIdxMap.containsKey(word2 + " " + word1)) {
						bitermToIdxMap.put(word2 + " " + word1, noOfBiterms);
						idxToBitermMap.put(noOfBiterms, word2 + " " + word1);
						noOfBiterms++;
					}
				}
			}
			words.add(wordList);
			documents.add(str);
			noOfDocs++;
		}
		in.close();
		
		pseudodocCount = new int[noOfPseudodocs];
		pseudodocSentimentTopicCount = new int[noOfPseudodocs][noOfSentiments][noOfTopics];
		pseudodocSentimentCount = new int[noOfPseudodocs][noOfSentiments];
		docSentimentTopicCount = new int[noOfDocs][noOfSentiments][noOfTopics];
		docSentimentCount = new int[noOfDocs][noOfSentiments];
		sentimentTopicWordCount = new int[noOfSentiments][noOfTopics][noOfWords];
		
		sumPseudodocCount = 0;
		sumPseudodocSentimentTopicCount = new int[noOfPseudodocs][noOfSentiments];
		sumPseudodocSentimentCount = new int[noOfPseudodocs];
		sumDocSentimentTopicCount = new int[noOfDocs][noOfSentiments];
		sumDocSentimentCount = new int[noOfDocs];
		sumSentimentTopicWordCount = new int[noOfSentiments][noOfTopics];
		
		multimatrix = new double[noOfSentiments][noOfTopics];
		for(int i=0; i<noOfSentiments; i++)
			for(int j=0; j<noOfTopics; j++)
				multimatrix[i][j] = 1.0 / (noOfSentiments*noOfTopics);
		
		multinomial = new double[noOfPseudodocs];
		for(int i=0; i<noOfPseudodocs; i++)
			multinomial[i] = 1.0 / noOfPseudodocs;
		
		topicMaps = new Vector<TreeMap<Integer, TreeMap<Integer, Integer>>>();
		sentimentMaps = new Vector<TreeMap<Integer, Integer>>();
		topics = new Vector<Vector<Integer>>();
		sentiments = new Vector<Vector<Integer>>();
		pseudodocs = new Vector<Integer>();
		for(int i=0; i<noOfDocs; i++) {
			int pse = nextDiscrete(multinomial);
			
			pseudodocCount[pse]++;
			sumPseudodocCount++;
			
			Vector<Integer> wordList = words.get(i);
			TreeMap<Integer, TreeMap<Integer, Integer>> topicMap = new TreeMap<Integer, TreeMap<Integer, Integer>>();
			for(int j=0; j<noOfSentiments; j++) {
				TreeMap<Integer, Integer> tMap = new TreeMap<Integer, Integer>();
				for(int k=0; k<noOfTopics; k++)
					tMap.put(k, 0);
				topicMap.put(j, tMap);
			}
			TreeMap<Integer, Integer> sentimentMap = new TreeMap<Integer, Integer>();
			for(int j=0; j<noOfSentiments; j++)
				sentimentMap.put(j, 0);
			Vector<Integer> topicList = new Vector<Integer>();
			Vector<Integer> sentimentList = new Vector<Integer>();
			
			for(int j=0; j<wordList.size(); j++) {
				for(int k=Math.max(0, j-window); k<=Math.min(wordList.size()-1, j+window); k++) {
					int word1 = wordList.get(j);
					int word2 = wordList.get(k);
					
					int[] arr = nextDiscrete(multimatrix);
					int sen = arr[0];
					int top = arr[1];
					
					int senWords = 0;
					for(int l=0; l<noOfSentiments; l++) {
						TreeSet<Integer> seed = seeds.get(l);
						if(seed.contains(word1) || seed.contains(word2)) {
							sen = l;
							senWords++;
							break;
						}
					}
					
					if(senWords > 1) sen = arr[0];
					
					pseudodocSentimentTopicCount[pse][sen][top]++;
					pseudodocSentimentCount[pse][sen]++;
					docSentimentTopicCount[i][sen][top]++;
					docSentimentCount[i][sen]++;
					sentimentTopicWordCount[sen][top][word1]++;
					sentimentTopicWordCount[sen][top][word2]++;
					
					sumPseudodocSentimentTopicCount[pse][sen]++;
					sumPseudodocSentimentCount[pse]++;
					sumDocSentimentTopicCount[i][sen]++;
					sumDocSentimentCount[i]++;
					sumSentimentTopicWordCount[sen][top]+=2;
					
					TreeMap<Integer, Integer> tMap = topicMap.get(sen);
					tMap.put(top, tMap.get(top)+1);
					sentimentMap.put(sen, sentimentMap.get(sen)+1);
					sentimentList.add(sen);
					topicList.add(top);
				}
			}
				
			sentiments.add(sentimentList);
			topics.add(topicList);
			topicMaps.add(topicMap);
			sentimentMaps.add(sentimentMap);
			pseudodocs.add(pse);
		}
		
		betaSums = new double[noOfSentiments];
		for(int i=0; i<noOfSentiments; i++)
			betaSums[i] = (seeds.get(i).size()*betas[2]) +
			              ((noOfWords-allSeeds.size())*betas[1]) +
			              ((allSeeds.size()-seeds.get(i).size())*betas[0]);
		
		for(int i=1; i<=noOfIters; i++) {
			System.out.println("iter: " + i + " ");
			
			for(int d=0; d<noOfDocs; d++) {
				int ps = pseudodocs.get(d);
				TreeMap<Integer, TreeMap<Integer, Integer>> topicMap = topicMaps.get(d);
				TreeMap<Integer, Integer> sentimentMap = sentimentMaps.get(d);
				
				Vector<Integer> wordList = words.get(d);
				Vector<Integer> topicList = topics.get(d);
				Vector<Integer> sentimentList = sentiments.get(d);
				
				pseudodocCount[ps]--;
				sumPseudodocCount--;
				
				for(Map.Entry<Integer, Integer> entry: sentimentMap.entrySet()) {
					int s = entry.getKey();
					int sc = entry.getValue();
					
					pseudodocSentimentCount[ps][s] -= sc;
					sumPseudodocSentimentCount[ps] -= sc;
					
					TreeMap<Integer, Integer> tMap = topicMap.get(s);
					
					for(Map.Entry<Integer, Integer> entry2 : tMap.entrySet()) {
						int t = entry2.getKey();
						int tc = entry2.getValue();
						
						pseudodocSentimentTopicCount[ps][s][t] -= tc;
						sumPseudodocSentimentTopicCount[ps][s] -= tc;
					}
				}
				
				for(int p=0; p<noOfPseudodocs; p++) {
					double cP = pseudodocCount[p] / (sumPseudodocCount + delta*noOfPseudodocs);
					double cPS = 1;
					
					double gamma0 = sumPseudodocSentimentCount[p] + gamma*noOfSentiments;
					int smi = 0;
					
					for(int s=0; s<noOfSentiments; s++) {
						int sc = sentimentMap.get(s);
						if(sc == 0) continue;
						
						double gammas = pseudodocSentimentCount[p][s] + gamma;
						
						for(int mis=0; mis<sc; mis++)
							cPS *= (gammas + mis) / (gamma0 + smi++);
					}
					
					double cPST = 1;
					
					for(int s=0; s<noOfSentiments; s++) {
						double alpha0 = sumPseudodocSentimentTopicCount[p][s] + alpha*noOfTopics;
						int stmi = 0;
						
						TreeMap<Integer, Integer> tMap = topicMap.get(s);
						
						for(int t=0; t<noOfTopics; t++) {
							int stc = tMap.get(t);
							if(stc == 0) continue;
							
							double alphast = pseudodocSentimentTopicCount[p][s][t] + alpha;
							
							for(int mist=0; mist<stc; mist++)
								cPST *= (alphast + mist) / (alpha0 + stmi++);
						}
					}
					
					multinomial[p] = cP * cPS * cPST;
				}
				
				pseudodocCount[ps]++;
				sumPseudodocCount++;
				
				for(Map.Entry<Integer, Integer> entry: sentimentMap.entrySet()) {
					int s = entry.getKey();
					int sc = entry.getValue();
					
					pseudodocSentimentCount[ps][s] += sc;
					sumPseudodocSentimentCount[ps] += sc;
					
					TreeMap<Integer, Integer> tMap = topicMap.get(s);
					
					for(Map.Entry<Integer, Integer> entry2 : tMap.entrySet()) {
						int t = entry2.getKey();
						int tc = entry2.getValue();
						
						pseudodocSentimentTopicCount[ps][s][t] += tc;
						sumPseudodocSentimentTopicCount[ps][s] += tc;
					}
				}
				
				pseudodocs.set(d, ps);

				int idx = 0;
				for(int j=0; j<wordList.size(); j++) {
					for(int k=Math.max(0, j-window); k<=Math.min(wordList.size()-1, j+window); k++) {
						int w1 = wordList.get(j);
						int w2 = wordList.get(k);
						int to = topicList.get(idx);
						int se = sentimentList.get(idx);
						
						pseudodocSentimentTopicCount[ps][se][to]--;
						pseudodocSentimentCount[ps][se]--;
						docSentimentTopicCount[d][se][to]--;
						docSentimentCount[d][se]--;
						sentimentTopicWordCount[se][to][w1]--;
						sentimentTopicWordCount[se][to][w2]--;
						
						sumPseudodocSentimentTopicCount[ps][se]--;
						sumPseudodocSentimentCount[ps]--;
						sumDocSentimentTopicCount[d][se]--;
						sumDocSentimentCount[d]--;
						sumSentimentTopicWordCount[se][to]-=2;
						
						TreeMap<Integer, Integer> tMap = topicMap.get(se);
						tMap.put(to, tMap.get(to)-1);
						sentimentMap.put(se, sentimentMap.get(se)-1);
	
						for(int s=0; s<noOfSentiments; s++)
							for(int t=0; t<noOfTopics; t++) {
								double cPST = (pseudodocSentimentTopicCount[ps][s][t] + alpha) / (sumPseudodocSentimentTopicCount[ps][s] + alpha*noOfTopics);
								double cPS = (pseudodocSentimentCount[ps][s] + gamma) / (sumPseudodocSentimentCount[ps] + gamma*noOfSentiments);
								
								double beta1 = betas[1];
								if(allSeeds.contains(w1)) {
									if(seeds.get(s).contains(w1)) beta1 = betas[2];
									else beta1 = betas[0];
								}
								double beta2 = betas[1];
								if(allSeeds.contains(w2)) {
									if(seeds.get(s).contains(w2)) beta2 = betas[2];
									else beta2 = betas[0];
								}
								
								double cSTW = ((sentimentTopicWordCount[s][t][w1] + beta1) * (sentimentTopicWordCount[s][t][w2] + beta2)) 
									    	/ ((sumSentimentTopicWordCount[s][t] + betaSums[s]) * (sumSentimentTopicWordCount[s][t] + betaSums[s] + 1));
							
								if((beta1 == betas[0] && beta2 == betas[2]) || (beta1 == betas[2] && beta1 == betas[0])) multimatrix[s][t] = 0;
								else multimatrix[s][t] = cPST * cPS * cSTW;
							}
						
						int[] arr = nextDiscrete(multimatrix);
						se = arr[0];
						to = arr[1];
						
						pseudodocSentimentTopicCount[ps][se][to]++;
						pseudodocSentimentCount[ps][se]++;
						docSentimentTopicCount[d][se][to]++;
						docSentimentCount[d][se]++;
						sentimentTopicWordCount[se][to][w1]++;
						sentimentTopicWordCount[se][to][w2]++;
						
						sumPseudodocSentimentTopicCount[ps][se]++;
						sumPseudodocSentimentCount[ps]++;
						sumDocSentimentTopicCount[d][se]++;
						sumDocSentimentCount[d]++;
						sumSentimentTopicWordCount[se][to]+=2;
						
						tMap = topicMap.get(se);
						tMap.put(to, tMap.get(to)+1);
						sentimentMap.put(se, sentimentMap.get(se)+1);
						topicList.set(idx, to);
						sentimentList.set(idx, se);
						
						idx++;
					}
				}
			}
		}
		
		write("data/mstm/");
	}
	
	public static void write(String dir) throws Exception {	
		new File(dir).mkdirs();
		PrintWriter printer;

		// pi (pseudodoc-topic)
		printer = new PrintWriter(new FileWriter(dir + filename + ".thetaPST"));
		for(int i=0; i<noOfSentiments; i++)
			for(int j=0; j<noOfTopics; j++)
				printer.print("\tSENTIMENT " + i + ", TOPIC " + j);
		printer.println();
		for(int i=0; i<noOfPseudodocs; i++) {
			printer.print("PSEUDODOC " + i);
			for(int j=0; j<noOfSentiments; j++)
				for(int k=0; k<noOfTopics; k++) {
					double pro = (pseudodocSentimentTopicCount[i][j][k] + alpha) / (sumPseudodocSentimentTopicCount[i][j] + alpha*noOfTopics);
					printer.print("\t" + pro);
				}
			printer.println();
		}
		printer.close();
		
		// pi (doc-sentiment)
		printer = new PrintWriter(new FileWriter(dir + filename + ".piDTS"));
		for(int j=0; j<noOfSentiments; j++)
			printer.print("\t(SENTIMENT " + j + ")");
		printer.println();
		for(int i=0; i<noOfDocs; i++) {
			String document = documents.get(i);
			printer.print(document);
			for(int j=0; j<noOfSentiments; j++) {
				double pro = (docSentimentCount[i][j] + gamma) / (sumDocSentimentCount[i] + gamma*noOfSentiments);
				printer.print("\t" + pro);
			}
			printer.println();
		}
		printer.close();
		
		// theta (doc-topic)
		printer = new PrintWriter(new FileWriter(dir + filename + ".thetaDST"));
		for(int i=0; i<noOfSentiments; i++)
			for(int j=0; j<noOfTopics; j++)
				printer.print("\tSENTIMENT" + i + ", TOPIC " + j);
		printer.println();
		for(int i=0; i<noOfDocs; i++) {
			String document = documents.get(i);
			printer.print(document);
			for(int j=0; j<noOfSentiments; j++)
				for(int k=0; k<noOfTopics; k++) {
					double pro = (docSentimentTopicCount[i][j][k] + alpha) / (sumDocSentimentTopicCount[i][j] + alpha*noOfTopics);
					printer.print("\t" + pro);
				}
			printer.println();
		}
		printer.close();

		// psi (doc-pseudodoc)
		printer = new PrintWriter(new FileWriter(dir + filename + ".psiDP"));
		for(int i=0; i<noOfDocs; i++) {
			String document = documents.get(i);
			int pseudodoc = pseudodocs.get(i);
			printer.println(document + "\t" + pseudodoc);
		}
		printer.close();
		
		// phi (sentiment-topic-word)
		String[][][] print = new String[noOfWords][noOfSentiments][noOfTopics];
		for(int i=0; i<noOfSentiments; i++)
			for(int j=0; j<noOfTopics; j++) {
				Map<Integer, Double> map = new TreeMap<Integer, Double>();
				for(int k=0; k<noOfWords; k++) {
					double beta = betas[1];
					if(allSeeds.contains(k)) {
						if(seeds.get(i).contains(k)) beta = betas[2];
						else beta = betas[0];
					}
					double pro = (sentimentTopicWordCount[i][j][k] + beta) / (sumSentimentTopicWordCount[i][j] + betaSums[i]);
					map.put(k, pro);
				}
				map = sort(map);
				int count = 0;
				for(Map.Entry<Integer, Double> entry : map.entrySet()) {
					String word = idxToWordMap.get(entry.getKey());
					print[count][i][j] = word + " (" + entry.getValue() + ")";
					count++;
//					if(count == 1000) break;
				}
			}
		printer = new PrintWriter(new FileWriter(dir + filename + ".phiSTW"));
		for(int i=0; i<noOfSentiments; i++)
			for(int j=0; j<noOfTopics; j++)
				printer.print("S" + i + ",T" + j + "\t");
		printer.println();
		for(int i=0; i<noOfWords; i++) {
			for(int j=0; j<noOfSentiments; j++)
				for(int k=0; k<noOfTopics; k++)
					printer.print(print[i][j][k] + "\t");
			printer.println();
		}
		printer.close();
	}

	public static int[] nextDiscrete(double[][] mult) {
		double sum = 0;
		for(int i=0; i<mult.length; i++)
			for(int j=0; j<mult[i].length; j++)
				sum += mult[i][j];
		double rand = Math.random()*sum;
		sum = 0;
		for(int i=0; i<mult.length; i++) {
			for(int j=0; j<mult[i].length; j++) {
				sum += mult[i][j];
				if(Double.compare(rand, sum) <= 0) {
					int[] arr = new int[2];
					arr[0] = i;
					arr[1] = j;
					return arr;
				}
			}
		}
		int[] arr = new int[2];
		arr[0] = mult.length-1;
		arr[1] = mult[0].length-1;
		return arr;
	}
	
	public static int nextDiscrete(double[] mult) {
		double sum = 0;
		for(int i=0; i<mult.length; i++)
			sum += mult[i];
		double rand = Math.random()*sum;
		sum = 0;
		for(int i=0; i<mult.length; i++) {
			sum += mult[i];
			if(Double.compare(rand, sum) <= 0)
				return i;
		}
		return mult.length-1;
	}
	
	public static <K, V extends Comparable<? super V>> Map<K, V> sort(Map<K, V> map) {
		List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
			public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
				return (o2.getValue()).compareTo(o1.getValue());
			}
		});

		Map<K, V> result = new LinkedHashMap<K, V>();
		for (int i=0; i<list.size(); i++)
			result.put(list.get(i).getKey(), list.get(i).getValue());
    
		return result;
	}
}
