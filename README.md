# Nomura-GMQuant-Challenge2022
My Solutions to Nomura's GM Quant Challenge 2022

The Nomura GM Quant Challenge was a 30-hour long contest held on 25th Jun 2022 - 26th Jun 2022. There were three questions in total which were graded as per how optimum your algorithm is. The PDF file for each question and my approach in .docx format is provided in each of the files. A brief overview will also be provided down below

*Question 1*
- The question was about a probabilty based financial model. Given the price of the product on trading day and some other parameters we had to find the value of the fair price of the product on trading day.
- The approach was fairly straight-forward. All we had to do was calculate all possible values of the value of fair price on the future date and backtrack to find the value on the trading day
- The second part was similar but longer in the sense that we needed to calculate the differentiation of fair price which was a slightly tedious calculation. Note that we weren't allowed to simply bump and revalue which was the simplest approach.

*Question 2*
- The question was to build an ideal trading algorithm to maximise the sharpe ratio of the algorithm. More on sharpe ratio can be found on the PDF

*Question 3*
- The question waws to build an IPL score predictor given a specific format of input.
- In the approach sheet I have explained my approach.
- One thing to note is that I have converted the training data to a pickle file, so in case you want to train it with a different set of data uncomment the commented part of the python file.