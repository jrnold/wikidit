Stvilla et al.

1.  Authority/Reputation
    - NumUniqueEditors
    - NumEdits
    - Connectivity: number of articles reachable through the editors
    - NumReverts
    - NumExternalLinks
    - NumRegUserEdits
    - NumAnonEdits
2. Completeness
    - NumBrokenWikilinks +
    - NumWikilinks
    - ArticleLength
3. Complexity
    - Flesch-Kincaid Readability Score
4. Informativeness
    - InfoNoise: proportion of text after removing media wiki code,
        stopwords, and stemming
    - Diversity = Num Unique editors / num eits
    - NumImages
5. Consistency
    - AdminEditShare
    - Age
6. Currency
    - Current article age in days
7. Volatility
    - Median revert time in minutes

Reverts to calculate Volatility, we applied
the approach of Priedhorsky et al., which uses regular
expressions to match edit comments [25]

Other features

-  number of references (<ref>)
-  number of headings
-  num headings
-  num references / article length
-  num images / article length
-  num wikilinks / articles length
-  num headings / article length
-  has infobox
-  num templates
-  num categories
-  tenuretime(t, i) = t - t(reg, i)
-  tenure edits(t, i) = log(nedits(i, now) * t / (t(now) - t(reg, i))
-  tenure(t, i) = tenuretime(t, i) = tenureedits(t, i)

The final actionable model is

1.  completeness = num broken wikilinks, num wikilinks
2.  informativeness = infonoise, num images
3.  num headings
4.  article length
5.  num references, aticle length

Visualizing structural completeness: https://wikiedu.org/blog/2016/09/16/visualizing-article-history-with-structural-completeness/

-  http://www.cs.cornell.edu/~cristian/Biased_language.html
-  https://web.stanford.edu/~jurafsky/pubs/neutrality.pdf
-  Detecting Biased Statements in Wikipedia (2018) Christoph Hube and Besnik Fetahu

Article Quality:

- Size Matters: Word Count as a Measure of Quality on Wikipedia. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.451.4208&rep=rep1&type=pdf

- Frequency counts
- character count complex word count sentence count
- token count one-syll. word count total syllables
- Readability indices
- Gunning fog index FORCAST formula Flesch-Kincaid
- Coleman-Liau Automatic Readability SMOG index
- Structural features
- internal links external links reference links
- category count image count reference count
- citation count table count section count

ORES Current Moodel

- revision characters
- content characters
- revision reference tags
- revision wikilinks
- revision external links
- headings by level (2)
- headings by level (3)
- image links
- category links
- cite templates
- ref_tags
- infobox templates
- cn templates
- who templates
- main article templates
- (stemmed lenth / content characters)
- log(revision paragraphs without refs to total length + 1)

- NPOV - neutral
- No non-free images
- NO original research
- No coyright violations

- More than 30-50kb or 4000 to 7000 words is too long [https://en.wikipedia.org/wiki/Wikipedia:Article_size]

  - > 100kb divided
  - > 60kb should be divided
  - > 50kb may need to be divided
  - < 40kb length does not justify
  - < 1kb

- Red links (page does not exists) none to file. DOn't have too many.
- DOn't link in lead sections
- Lead section length:

  - < 15000 charadcters: 1-2 paragraphs
  - 15000-30000 characters, 2-3 paragraphs
  - 30000+ characters 3-4 paragraphs

-  https://www.researchgate.net/profile/Wlodzimierz_Lewoniewski/publication/308887798_Quality_and_Importance_of_Wikipedia_Articles_in_Different_Languages/links/59ef4c9e458515ec0c7b5ad2/Quality-and-Importance-of-Wikipedia-Articles-in-Different-Languages.pdf
- Does article importance affect its quality?
– What parameters can help to assess the importance of the article automatically?
– Is there a difference between importance models in different languages?
- https://www.researchgate.net/profile/Wlodzimierz_Lewoniewski/publication/308887798_Quality_and_Importance_of_Wikipedia_Articles_in_Different_Languages/links/59ef4c9e458515ec0c7b5ad2/Quality-and-Importance-of-Wikipedia-Articles-in-Different-Languages.pdf


- Wipedia Article Finder for WikiEdu
- [Suggest Bot](https://en.wikipedia.org/wiki/User:SuggestBot): uses ORES to predict
- Use ORES for articles https://dashboard-testing.wikiedu.org/courses/test/ORES_playground
- Objective Revision (https://ores.wikimedia.org/)
- https://blog.wikimedia.org/2015/11/30/artificial-intelligence-x-ray-specs/


- Structuring Wikipedia Articles with Section Recommendations Tiziano Piccardi, Michele Catasta, Leila Zia, Robert West - https://arxiv.org/abs/1804.05995
- PetScan to search wikipedia
- ORES in python
- Bad words: https://www.mediawiki.org/wiki/ORES/BWDS_review
- https://www.mediawiki.org/wiki/ORES/Scholarship
- ell Me More: An Actionable Quality Model for Wikipedia: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.739.4279&rep=rep1&type=pdf
- Identifying Semantic Edit Intentions from Revisions in Wikipedia: http://www.cs.cmu.edu/~diyiy/docs/emnlp17.pdf
- Diyi Yang, Aaron Halfaker, Robert Kraut, and Eduard
Hovy. 2016. Who did what: Editor role identification
in wikipedia. In Tenth International AAAI
Conference on Web and Social Media.
