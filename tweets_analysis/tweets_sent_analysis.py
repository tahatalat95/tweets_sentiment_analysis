{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9483d64a",
   "metadata": {},
   "source": [
    "\n",
    "# libraries needed "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "989f3c88",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install nltk\n",
    "#!pip install seaborn   # used for virtualization\n",
    "#!pip install wordcloud\n",
    "#!pip install gensim    # converting data into features "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b89e6ac3",
   "metadata": {},
   "source": [
    "# import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "82f592d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re               #for regular expressions\n",
    "import nltk             #for text manipulation\n",
    "import string\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b699b285",
   "metadata": {},
   "source": [
    "# Set Some Options"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "078cc006",
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.set_option(\"display.max_colwidth\",200) # makes the maximim column width equal to 200 unit for displaying the dataset\n",
    "warnings.filterwarnings(\"ignore\",category=DeprecationWarning) # ignore any deprecated features in dataset\n",
    "%matplotlib inline  \n",
    "#magic function display inline in jupyter instead of being displayed in another windows"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ee1ac7a",
   "metadata": {},
   "source": [
    "# Reading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "684eb504",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('train.csv') # read .csv training dataset\n",
    "test = pd.read_csv('test.csv')   # read .csv test dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "abd35bd1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>14</td>\n",
       "      <td>1</td>\n",
       "      <td>@user #cnn calls #michigan middle school 'build the wall' chant '' #tcot</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>15</td>\n",
       "      <td>1</td>\n",
       "      <td>no comment!  in #australia   #opkillingbay #seashepherd #helpcovedolphins #thecove  #helpcovedolphins</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>retweet if you agree!</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>24</td>\n",
       "      <td>1</td>\n",
       "      <td>@user @user lumpy says i am a . prove it lumpy.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>35</td>\n",
       "      <td>1</td>\n",
       "      <td>it's unbelievable that in the 21st century we'd need something like this. again. #neverump  #xenophobia</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>56</th>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>@user lets fight against  #love #peace</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68</th>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>ð©the white establishment can't have blk folx running around loving themselves and promoting our greatness</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77</th>\n",
       "      <td>78</td>\n",
       "      <td>1</td>\n",
       "      <td>@user hey, white people: you can call people 'white' by @user  #race  #identity #medâ¦</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>82</th>\n",
       "      <td>83</td>\n",
       "      <td>1</td>\n",
       "      <td>how the #altright uses  &amp;amp; insecurity to lure men into #whitesupremacy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>112</td>\n",
       "      <td>1</td>\n",
       "      <td>@user i'm not interested in a #linguistics that doesn't address #race &amp;amp; . racism is about #power. #raciolinguistics bringsâ¦</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  label  \\\n",
       "13    14      1   \n",
       "14    15      1   \n",
       "17    18      1   \n",
       "23    24      1   \n",
       "34    35      1   \n",
       "56    57      1   \n",
       "68    69      1   \n",
       "77    78      1   \n",
       "82    83      1   \n",
       "111  112      1   \n",
       "\n",
       "                                                                                                                                 tweet  \n",
       "13                                                          @user #cnn calls #michigan middle school 'build the wall' chant '' #tcot    \n",
       "14                               no comment!  in #australia   #opkillingbay #seashepherd #helpcovedolphins #thecove  #helpcovedolphins  \n",
       "17                                                                                                              retweet if you agree!   \n",
       "23                                                                                     @user @user lumpy says i am a . prove it lumpy.  \n",
       "34                            it's unbelievable that in the 21st century we'd need something like this. again. #neverump  #xenophobia   \n",
       "56                                                                                             @user lets fight against  #love #peace   \n",
       "68                      ð©the white establishment can't have blk folx running around loving themselves and promoting our greatness    \n",
       "77                                             @user hey, white people: you can call people 'white' by @user  #race  #identity #medâ¦  \n",
       "82                                                       how the #altright uses  &amp; insecurity to lure men into #whitesupremacy      \n",
       "111  @user i'm not interested in a #linguistics that doesn't address #race &amp; . racism is about #power. #raciolinguistics bringsâ¦  "
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[train['label'] == 1].head(10)  # display racist train dataset (first 10 rows) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "b3a958cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>@user camping tomorrow @user @user @user @user @user @user @user dannyâ¦</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>@user @user welcome here !  i'm   it's so #gr8 !</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1      0   \n",
       "1   2      0   \n",
       "2   3      0   \n",
       "3   4      0   \n",
       "4   5      0   \n",
       "5   6      0   \n",
       "6   7      0   \n",
       "7   8      0   \n",
       "8   9      0   \n",
       "9  10      0   \n",
       "\n",
       "                                                                                                                                             tweet  \n",
       "0                                            @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run  \n",
       "1                       @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked  \n",
       "2                                                                                                                              bihday your majesty  \n",
       "3                                                           #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦    \n",
       "4                                                                                                           factsguide: society now    #motivation  \n",
       "5                             [2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo    \n",
       "6                                                                        @user camping tomorrow @user @user @user @user @user @user @user dannyâ¦  \n",
       "7  the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl  \n",
       "8                                                          we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦   \n",
       "9                                                                                                @user @user welcome here !  i'm   it's so #gr8 !   "
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[train['label'] == 0].head(10)  # display non-racist train dataset (first 10 rows) [غير عنصري]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "ba67a3e7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(31962, 3)\n",
      "(17197, 2)\n"
     ]
    }
   ],
   "source": [
    "print (train.shape)\n",
    "print (test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "7df8642e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.pie(train['label'].value_counts(),labels=['0','1'],autopct='%1.1f%%')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "7bfaaa2a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    29720\n",
       "1     2242\n",
       "Name: label, dtype: int64"
      ]
     },
     "execution_count": 161,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['label'].value_counts() # counts the number of non-racist rocords which has value of [0] and \n",
    "# number of racist records which has value of [1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "f7a8e860",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "length_train_dataset = train['tweet'].str.len() # length of characters of every tweet in train dataset tweets\n",
    "length_test_dataset = test['tweet'].str.len()   # length of characters of every tweet in test dataset tweets\n",
    "\n",
    "plt.hist(length_train_dataset, bins=20, label=\"Train tweets\") # create histogram for train tweets with possible 20 columns\n",
    "plt.hist(length_test_dataset, bins=20, label=\"Test tweets\")   # create histogram for test tweets with possible 20 columns\n",
    "plt.legend() # make two different colors for two datasets\n",
    "plt.show()\n",
    "# x-axis represnts number of characters in every tweet\n",
    "# y-axis represents number of tweets represent that number of characters"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce81b768",
   "metadata": {},
   "source": [
    "# concatnating data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "719eb00b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(49159, 3)"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combine = pd.concat([train,test],ignore_index=True) # combine train and test datasets into one dataset called (combine) And ignore index\n",
    "combine.shape # display (rows,columns) for combine dataset\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e448504",
   "metadata": {},
   "source": [
    "# define function to remove specific pattern"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "eafbd25d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_pattern(input_text,pattern):\n",
    "    r = re.findall(pattern, input_text) \n",
    "    # store in (r) every thing in input_text that match the pattern\n",
    "    # information about findall() method => https://www.w3schools.com/python/python_regex.asp#findall\n",
    "    for i in r:\n",
    "        input_text = re.sub(i,'',input_text)\n",
    "        # replace every word => [i] with '' to remove it from dataset \n",
    "        # view information about sub() method\n",
    "    return input_text"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d7deb41",
   "metadata": {},
   "source": [
    "\n",
    "# removing twitter handle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "be3c8247",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "      <th>raw_tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "      <td>when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "      <td>thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "      <td>bihday your majesty</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1    0.0   \n",
       "1   2    0.0   \n",
       "2   3    0.0   \n",
       "3   4    0.0   \n",
       "4   5    0.0   \n",
       "\n",
       "                                                                                                                        tweet  \\\n",
       "0                       @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run   \n",
       "1  @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked   \n",
       "2                                                                                                         bihday your majesty   \n",
       "3                                      #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦     \n",
       "4                                                                                      factsguide: society now    #motivation   \n",
       "\n",
       "                                                                                                          raw_tweet  \n",
       "0                   when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run  \n",
       "1    thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked  \n",
       "2                                                                                               bihday your majesty  \n",
       "3                            #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦    \n",
       "4                                                                            factsguide: society now    #motivation  "
      ]
     },
     "execution_count": 165,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combine['raw_tweet'] = np.vectorize(remove_pattern)(combine['tweet'],\"@[\\w]*\") # w-> set of chars \\ -> sepcial sequence # any number of occurrences\n",
    "# remove any (@) followed by any number of charachers until space \n",
    "# (like this tweet: ( @ahmed I am happy ), it results into => ( I am happy) )\n",
    "# and store it in column called 'tidy_tweet'\n",
    "# we used vectorize (loop alternative) because if there many objects (it is not single tweet)\n",
    "combine.head() # display first 5 records"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ee6fc4e",
   "metadata": {},
   "source": [
    "# removing links"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "572b1a47",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "      <th>raw_tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "      <td>when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "      <td>thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "      <td>bihday your majesty</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1    0.0   \n",
       "1   2    0.0   \n",
       "2   3    0.0   \n",
       "3   4    0.0   \n",
       "4   5    0.0   \n",
       "\n",
       "                                                                                                                        tweet  \\\n",
       "0                       @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run   \n",
       "1  @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked   \n",
       "2                                                                                                         bihday your majesty   \n",
       "3                                      #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦     \n",
       "4                                                                                      factsguide: society now    #motivation   \n",
       "\n",
       "                                                                                                          raw_tweet  \n",
       "0                   when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run  \n",
       "1    thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked  \n",
       "2                                                                                               bihday your majesty  \n",
       "3                            #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦    \n",
       "4                                                                            factsguide: society now    #motivation  "
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combine['raw_tweet'] = np.vectorize(remove_pattern)(combine['raw_tweet'],r\"http?:\\/\\/S+\") # w-> set of chars \\ -> sepcial sequence # any number of occurrences\n",
    "combine.head() # display first 5 records"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53c4de23",
   "metadata": {},
   "source": [
    "# replace any sequence except ( a-z ) ( A-Z ) ( # ) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "0b40b480",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "      <th>raw_tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "      <td>when a father is dysfunctional and is so selfish he drags his kids into his dysfunction    #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "      <td>thanks for #lyft credit i can t use cause they don t offer wheelchair vans in pdx     #disapointed #getthanked</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "      <td>bihday your majesty</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "      <td>#model   i love u take with u all the time in ur</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "      <td>factsguide  society now    #motivation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo</td>\n",
       "      <td>huge fan fare and big talking before they leave  chaos and pay disputes when they get there  #allshowandnogo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user camping tomorrow @user @user @user @user @user @user @user dannyâ¦</td>\n",
       "      <td>camping tomorrow        danny</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl</td>\n",
       "      <td>the next school year is the year for exams      can t think about that      #school #exams   #hate #imagine #actorslife #revolutionschool #girl</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦</td>\n",
       "      <td>we won    love the land    #allin #cavs #champions #cleveland #clevelandcavaliers</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user welcome here !  i'm   it's so #gr8 !</td>\n",
       "      <td>welcome here    i m   it s so #gr</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1    0.0   \n",
       "1   2    0.0   \n",
       "2   3    0.0   \n",
       "3   4    0.0   \n",
       "4   5    0.0   \n",
       "5   6    0.0   \n",
       "6   7    0.0   \n",
       "7   8    0.0   \n",
       "8   9    0.0   \n",
       "9  10    0.0   \n",
       "\n",
       "                                                                                                                                             tweet  \\\n",
       "0                                            @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run   \n",
       "1                       @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked   \n",
       "2                                                                                                                              bihday your majesty   \n",
       "3                                                           #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦     \n",
       "4                                                                                                           factsguide: society now    #motivation   \n",
       "5                             [2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo     \n",
       "6                                                                        @user camping tomorrow @user @user @user @user @user @user @user dannyâ¦   \n",
       "7  the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl   \n",
       "8                                                          we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦    \n",
       "9                                                                                                @user @user welcome here !  i'm   it's so #gr8 !    \n",
       "\n",
       "                                                                                                                                         raw_tweet  \n",
       "0                                                  when a father is dysfunctional and is so selfish he drags his kids into his dysfunction    #run  \n",
       "1                                   thanks for #lyft credit i can t use cause they don t offer wheelchair vans in pdx     #disapointed #getthanked  \n",
       "2                                                                                                                              bihday your majesty  \n",
       "3                                                           #model   i love u take with u all the time in ur                                        \n",
       "4                                                                                                           factsguide  society now    #motivation  \n",
       "5                                   huge fan fare and big talking before they leave  chaos and pay disputes when they get there  #allshowandnogo    \n",
       "6                                                                                                                 camping tomorrow        danny     \n",
       "7  the next school year is the year for exams      can t think about that      #school #exams   #hate #imagine #actorslife #revolutionschool #girl  \n",
       "8                                                          we won    love the land    #allin #cavs #champions #cleveland #clevelandcavaliers        \n",
       "9                                                                                                            welcome here    i m   it s so #gr      "
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combine['raw_tweet'] = combine['raw_tweet'].str.replace(\"[^a-zA-Z#]\",\" \",regex=True)\n",
    "# This replace any sequence of character EXCEPT (A-Z) or (a-z) or # to white space\n",
    "# like (!?&^ahmed) => (ahmed) \n",
    "combine.head(10) # display first 10 records"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e4b7ef5",
   "metadata": {},
   "source": [
    "# removing stopwords"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "639a62dc",
   "metadata": {},
   "source": [
    "# first method ( not efficent )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "c8356b63",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#combine['tidy_tweet'] = combine['tidy_tweet'].apply(lambda x : ' '.join([w for w in x.split() if len(w) > 3]))\n",
    "# first it joins words in the tweet an separate them with space\n",
    "# but while joining it applies a restrction to prevent words that have length less than 4 to joined  \n",
    "# removing words which length is 3 or less\n",
    "# read about lambda function in python https://www.w3schools.com/python/python_lambda.asp\n",
    "#combine.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "438969c2",
   "metadata": {},
   "source": [
    "# removing stopwords second method (better)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "bd14c629",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "stopwords=set(stopwords.words('english'))\n",
    "\n",
    " \n",
    "def remove_stopwords(tweets):\n",
    "    all_tweets=[]              # to carry all no_stopwords tweets \n",
    "    for tweet in tweets:\n",
    "        no_sw_list=[]          # to carry all non_stopwords on the tweet \n",
    "        for word in tweet.split():  # spliting the tweet \n",
    "            if word.lower() not in stopwords: # comparing words on the tweet to words on english stopwords \n",
    "                 no_sw_list.append(word) # adding non_stopwords in the tweet to no_sw_list \n",
    "        all_tweets.append(' '.join(no_sw_list)) # adding the tweet( after removing stopwords ) to the tweets array \n",
    "    return   all_tweets\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "847ce8c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "      <th>raw_tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "      <td>father dysfunctional selfish drags kids dysfunction #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "      <td>thanks #lyft credit use cause offer wheelchair vans pdx #disapointed #getthanked</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "      <td>bihday majesty</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "      <td>#model love u take u time ur</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "      <td>factsguide society #motivation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo</td>\n",
       "      <td>huge fan fare big talking leave chaos pay disputes get #allshowandnogo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user camping tomorrow @user @user @user @user @user @user @user dannyâ¦</td>\n",
       "      <td>camping tomorrow danny</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl</td>\n",
       "      <td>next school year year exams think #school #exams #hate #imagine #actorslife #revolutionschool #girl</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦</td>\n",
       "      <td>love land #allin #cavs #champions #cleveland #clevelandcavaliers</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user welcome here !  i'm   it's so #gr8 !</td>\n",
       "      <td>welcome #gr</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1    0.0   \n",
       "1   2    0.0   \n",
       "2   3    0.0   \n",
       "3   4    0.0   \n",
       "4   5    0.0   \n",
       "5   6    0.0   \n",
       "6   7    0.0   \n",
       "7   8    0.0   \n",
       "8   9    0.0   \n",
       "9  10    0.0   \n",
       "\n",
       "                                                                                                                                             tweet  \\\n",
       "0                                            @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run   \n",
       "1                       @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked   \n",
       "2                                                                                                                              bihday your majesty   \n",
       "3                                                           #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦     \n",
       "4                                                                                                           factsguide: society now    #motivation   \n",
       "5                             [2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo     \n",
       "6                                                                        @user camping tomorrow @user @user @user @user @user @user @user dannyâ¦   \n",
       "7  the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl   \n",
       "8                                                          we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦    \n",
       "9                                                                                                @user @user welcome here !  i'm   it's so #gr8 !    \n",
       "\n",
       "                                                                                             raw_tweet  \n",
       "0                                             father dysfunctional selfish drags kids dysfunction #run  \n",
       "1                     thanks #lyft credit use cause offer wheelchair vans pdx #disapointed #getthanked  \n",
       "2                                                                                       bihday majesty  \n",
       "3                                                                         #model love u take u time ur  \n",
       "4                                                                       factsguide society #motivation  \n",
       "5                               huge fan fare big talking leave chaos pay disputes get #allshowandnogo  \n",
       "6                                                                               camping tomorrow danny  \n",
       "7  next school year year exams think #school #exams #hate #imagine #actorslife #revolutionschool #girl  \n",
       "8                                     love land #allin #cavs #champions #cleveland #clevelandcavaliers  \n",
       "9                                                                                          welcome #gr  "
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tweets=combine['raw_tweet']\n",
    "combine['raw_tweet']=remove_stopwords(tweets) \n",
    "combine.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "81668183",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                                                     [father, dysfunctional, selfish, drags, kids, dysfunction, #run]\n",
       "1                         [thanks, #lyft, credit, use, cause, offer, wheelchair, vans, pdx, #disapointed, #getthanked]\n",
       "2                                                                                                    [bihday, majesty]\n",
       "3                                                                                 [#model, love, u, take, u, time, ur]\n",
       "4                                                                                   [factsguide, society, #motivation]\n",
       "5                                   [huge, fan, fare, big, talking, leave, chaos, pay, disputes, get, #allshowandnogo]\n",
       "6                                                                                           [camping, tomorrow, danny]\n",
       "7    [next, school, year, year, exams, think, #school, #exams, #hate, #imagine, #actorslife, #revolutionschool, #girl]\n",
       "8                                             [love, land, #allin, #cavs, #champions, #cleveland, #clevelandcavaliers]\n",
       "9                                                                                                       [welcome, #gr]\n",
       "Name: raw_tweet, dtype: object"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# tokenise means that [bihday your majesty] => [bihday , your , majesty]\n",
    "tokenized_tweet = combine['raw_tweet'].apply(lambda x : x.split()) # tokenise the words\n",
    "tokenized_tweet.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abcbad7f",
   "metadata": {},
   "source": [
    "# stem all words like (kids to kid) (society to societi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "36950e02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                                                                              [father, dysfunct, selfish, drag, kid, dysfunct, #run]\n",
       "1                                               [thank, #lyft, credit, use, caus, offer, wheelchair, van, pdx, #disapoint, #getthank]\n",
       "2                                                                                                                   [bihday, majesti]\n",
       "3                                                                                                [#model, love, u, take, u, time, ur]\n",
       "4                                                                                                        [factsguid, societi, #motiv]\n",
       "                                                                     ...                                                             \n",
       "49154                              [thought, factori, left, right, polaris, #trump, #uselect, #leadership, #polit, #brexit, #blm, gt]\n",
       "49155                                           [feel, like, mermaid, #hairflip, #neverreadi, #formal, #wed, #gown, #dress, #mermaid]\n",
       "49156    [#hillari, #campaign, today, #ohio, omg, amp, use, word, like, asset, amp, liabil, never, #clinton, say, thee, word, #radic]\n",
       "49157                                            [happi, work, confer, right, mindset, lead, cultur, develop, organ, #work, #mindset]\n",
       "49158                                                                       [song, glad, free, download, #shoegaz, #newmus, #newsong]\n",
       "Name: raw_tweet, Length: 49159, dtype: object"
      ]
     },
     "execution_count": 172,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from nltk.stem.porter import *\n",
    "stemmer = PorterStemmer()\n",
    "tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) \n",
    "# apply stemming process to all words on each x (tweet)\n",
    "tokenized_tweet"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3876ea24",
   "metadata": {},
   "source": [
    "# now let's combine these tokens back"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "1b00a9b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "      <th>raw_tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "      <td>father dysfunct selfish drag kid dysfunct #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "      <td>thank #lyft credit use caus offer wheelchair van pdx #disapoint #getthank</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "      <td>bihday majesti</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "      <td>#model love u take u time ur</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "      <td>factsguid societi #motiv</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1    0.0   \n",
       "1   2    0.0   \n",
       "2   3    0.0   \n",
       "3   4    0.0   \n",
       "4   5    0.0   \n",
       "\n",
       "                                                                                                                        tweet  \\\n",
       "0                       @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run   \n",
       "1  @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked   \n",
       "2                                                                                                         bihday your majesty   \n",
       "3                                      #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦     \n",
       "4                                                                                      factsguide: society now    #motivation   \n",
       "\n",
       "                                                                   raw_tweet  \n",
       "0                             father dysfunct selfish drag kid dysfunct #run  \n",
       "1  thank #lyft credit use caus offer wheelchair van pdx #disapoint #getthank  \n",
       "2                                                             bihday majesti  \n",
       "3                                               #model love u take u time ur  \n",
       "4                                                   factsguid societi #motiv  "
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "#  like [bihday, your, majesti] => [bihday your majesti]\n",
    "for i in range(len(tokenized_tweet)):\n",
    "    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) #concat all words into one sentence\n",
    "# for each token set(i) join all tokens back again \n",
    "combine['raw_tweet'] = tokenized_tweet\n",
    "combine.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca11ff9a",
   "metadata": {},
   "source": [
    "# Creating word cloud for only visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "27d71144",
   "metadata": {},
   "outputs": [],
   "source": [
    "#all_words = ' '.join([text for text in combine['tidy_tweet']]) # combine all tweets separated by space \n",
    "#from wordcloud import WordCloud\n",
    "#wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words) # draw wordcloud\n",
    "#plt.figure(figsize=(10,7))  # figure dimentions \n",
    "#plt.imshow(wordcloud, interpolation=\"bilinear\")\n",
    "# pixels interpolation \n",
    "# about interpolation ->> https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/interpolation_methods.html \n",
    "#plt.axis('off') # disable axis\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "431386fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "#non-rasist\n",
    "#normal_words= ' '.join([text for text in combine['tidy_tweet'][combine['label']==0]])\n",
    "#wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)\n",
    "#plt.figure(figsize=(10,7))\n",
    "#plt.imshow(wordcloud,interpolation='bilinear')\n",
    "#plt.axis('off')\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "44916e70",
   "metadata": {},
   "outputs": [],
   "source": [
    "#rasist tweet\n",
    "#negative_words= ' '.join([text for text in combine['tidy_tweet'][combine['label']==1]])\n",
    "#wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)\n",
    "#plt.figure(figsize=(10,7))\n",
    "#plt.imshow(wordcloud,interpolation='bilinear')\n",
    "#plt.axis('off')\n",
    "#plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "c60806de",
   "metadata": {},
   "outputs": [],
   "source": [
    "# collect hashtags\n",
    "def hashtag_extract(tweets):\n",
    "    hashtag_list = []\n",
    "    for tweet in tweets: # loop over words contained in tweet\n",
    "        ht = re.findall(r\"#(\\w+)\",tweet) # find a pattern that begins with hash sign (#) until finding a space\n",
    "        hashtag_list.append(ht) # append hashtags to (hashtags) list\n",
    "    return hashtag_list\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "aa47a683",
   "metadata": {},
   "outputs": [],
   "source": [
    "# extracting hashtags from non racist tweets\n",
    "hash_regular=hashtag_extract(combine['raw_tweet'][combine['label']==0])\n",
    "hash_regular=sum(hash_regular,[]) # list of all non racist hashtags\n",
    "\n",
    "# extracting hashtags from racist tweets\n",
    "hash_racist=hashtag_extract(combine['raw_tweet'][combine['label']==1])\n",
    "hash_racist=sum(hash_racist,[])# list of all racist hashtags\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "a88f5b3b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABa8AAAJNCAYAAAA21omXAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA+c0lEQVR4nO3de7hkV1kn4N9Hmvu1YxoMCdiRCWJARWkYrgriADpqUEGCFxJFM1wEyYiMGWcg6mQGhTEOOqARMUEzwYAgEeQaCfcQGgjkAoEMCSSQIUEigkgwYc0fe5905aTO6dPpU1Wrut/3ec5zdq3aVfVV1a61d/1q1apqrQUAAAAAAHpyi0UXAAAAAAAAqwmvAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6M6WRRcwKwcddFDbvn37ossAAAAAAGAdH/rQh77YWtu2un2fDa+3b9+enTt3LroMAAAAAADWUVWfmdZu2hAAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7WxZdwLxsPW7rokuY6pqTrll0CQAAAAAA3THyGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6M7MwuuqekVVXVVVF6xqf1ZVXVxVF1bV7020H19Vl4znPXai/QFVdf543kuqqmZVMwAAAAAAfZjlyOtTkjxusqGqHpXkyCTf3Vq7b5IXj+1HJDkqyX3Hy7y0qg4YL/ayJMcmOXz8u9F1AgAAAACw75lZeN1ae1eSL61qfnqSF7bWrh3XuWpsPzLJq1pr17bWLk1ySZIHVdXBSe7UWnt/a60leWWSx8+qZgAAAAAA+jDvOa/vneQRVfWBqnpnVT1wbD8kyeUT610xth0yLq9uBwAAAABgH7ZlAbe3NcmDkzwwyRlV9e1Jps1j3dZpn6qqjs0wxUjuec977nWxAAAAAAAsxrxHXl+R5LVtcG6SbyY5aGy/x8R6hyb5/Nh+6JT2qVprJ7fWdrTWdmzbtm3TiwcAAAAAYD7mHV7/TZIfTJKquneSWyX5YpIzkxxVVbeuqsMy/DDjua21K5N8paoeXFWV5ClJXj/nmgEAAAAAmLOZTRtSVacneWSSg6rqiiQvSPKKJK+oqguSfCPJ0eMPMV5YVWckuSjJdUme2Vq7fryqpyc5Jcltk7xp/AMAAAAAYB82s/C6tfbkNc76uTXWPzHJiVPadya53yaWBgAAAABA5+Y9bQgAAAAAAOyW8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuzCy8rqpXVNVVVXXBlPOeW1Wtqg6aaDu+qi6pqour6rET7Q+oqvPH815SVTWrmgEAAAAA6MMsR16fkuRxqxur6h5J/l2Sz060HZHkqCT3HS/z0qo6YDz7ZUmOTXL4+HeT6wQAAAAAYN8ys/C6tfauJF+actZJSZ6XpE20HZnkVa21a1trlya5JMmDqurgJHdqrb2/tdaSvDLJ42dVMwAAAAAAfZjrnNdV9eNJPtda++iqsw5JcvnE6SvGtkPG5dXtAAAAAADsw7bM64aq6nZJfjPJY6adPaWtrdO+1m0cm2GKkdzznve8GVUCAAAAANCDeY68vleSw5J8tKouS3Jokg9X1bdmGFF9j4l1D03y+bH90CntU7XWTm6t7Wit7di2bdsmlw8AAAAAwLzMLbxurZ3fWrtra217a217hmD6+1pr/y/JmUmOqqpbV9VhGX6Y8dzW2pVJvlJVD66qSvKUJK+fV80AAAAAACzGzMLrqjo9yfuTfEdVXVFVT11r3dbahUnOSHJRkjcneWZr7frx7KcneXmGH3H8v0neNKuaAQAAAADow8zmvG6tPXk3529fdfrEJCdOWW9nkvttanEAAAAAAHRtnnNeAwAAAADAhgivAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO7MLLyuqldU1VVVdcFE24uq6hNV9bGqel1V3WXivOOr6pKquriqHjvR/oCqOn887yVVVbOqGQAAAACAPsxy5PUpSR63qu1tSe7XWvvuJJ9McnySVNURSY5Kct/xMi+tqgPGy7wsybFJDh//Vl8nAAAAAAD7mJmF1621dyX50qq2t7bWrhtPnpPk0HH5yCSvaq1d21q7NMklSR5UVQcnuVNr7f2ttZbklUkeP6uaAQAAAADowyLnvP7FJG8alw9JcvnEeVeMbYeMy6vbAQAAAADYhy0kvK6q30xyXZLTVpqmrNbWaV/reo+tqp1VtfPqq6/e+0IBAAAAAFiIuYfXVXV0kh9N8rPjVCDJMKL6HhOrHZrk82P7oVPap2qtndxa29Fa27Ft27bNLRwAAAAAgLmZa3hdVY9L8p+S/Hhr7WsTZ52Z5KiqunVVHZbhhxnPba1dmeQrVfXgqqokT0ny+nnWDAAAAADA/G2Z1RVX1elJHpnkoKq6IskLkhyf5NZJ3jZk0Tmntfa01tqFVXVGkosyTCfyzNba9eNVPT3JKUlum2GO7DcFAAAAAIB92szC69bak6c0/9k665+Y5MQp7TuT3G8TSwMAAAAAoHML+cFGAAAAAABYj/AaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7swsvK6qV1TVVVV1wUTbgVX1tqr61Ph/68R5x1fVJVV1cVU9dqL9AVV1/njeS6qqZlUzAAAAAAB9mOXI61OSPG5V228kOau1dniSs8bTqaojkhyV5L7jZV5aVQeMl3lZkmOTHD7+rb5OAAAAAAD2MTMLr1tr70rypVXNRyY5dVw+NcnjJ9pf1Vq7trV2aZJLkjyoqg5OcqfW2vtbay3JKycuAwAAAADAPmrec17frbV2ZZKM/+86th+S5PKJ9a4Y2w4Zl1e3AwAAAACwD+vlBxunzWPd1mmffiVVx1bVzqraefXVV29acQAAAAAAzNe8w+svjFOBZPx/1dh+RZJ7TKx3aJLPj+2HTmmfqrV2cmttR2ttx7Zt2za1cAAAAAAA5mfe4fWZSY4el49O8vqJ9qOq6tZVdViGH2Y8d5xa5CtV9eCqqiRPmbgMAAAAAAD7qC2zuuKqOj3JI5McVFVXJHlBkhcmOaOqnprks0memCSttQur6owkFyW5LskzW2vXj1f19CSnJLltkjeNfwAAAAAA7MNmFl631p68xlmPXmP9E5OcOKV9Z5L7bWJpAAAAAAB0rpcfbAQAAAAAgBsIrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7mxZdAHs3tbjti66hJu45qRrFl0CAAAAALAPM/IaAAAAAIDubCi8rqqHbaQNAAAAAAA2w0ZHXv/hBtsAAAAAAGCvrTvndVU9JMlDk2yrqv84cdadkhwwy8IAAAAAANh/7e4HG2+V5A7jenecaP+nJE+YVVEAAAAAAOzf1g2vW2vvTPLOqjqltfaZOdUEAAAAAMB+bncjr1fcuqpOTrJ98jKttR+cRVEAAAAAAOzfNhpevzrJHyd5eZLrZ1cOAAAAAABsPLy+rrX2splWAgAAAAAAo1tscL2/rapnVNXBVXXgyt9MKwMAAAAAYL+10ZHXR4//f32irSX59s0tBwAAAAAANhhet9YOm3UhAAAAAACwYkPhdVU9ZVp7a+2Vm1sOAAAAAABsfNqQB04s3ybJo5N8OInwGgAAAACATbfRaUOeNXm6qu6c5C9mUhEAAAAAAPu9W9zMy30tyeGbWQgAAAAAAKzY6JzXf5ukjScPSPKdSc6YVVEAAAAAAOzfNjrn9Ysnlq9L8pnW2hUzqAcAAAAAADY2bUhr7Z1JPpHkjkm2JvnGLIsCAAAAAGD/tqHwuqp+Osm5SZ6Y5KeTfKCqnjDLwgAAAAAA2H9tdNqQ30zywNbaVUlSVduSvD3Ja2ZVGAAAAAAA+68NjbxOcouV4Hr0D3twWQAAAAAA2CMbHXn95qp6S5LTx9NPSvJ3sykJAAAAAID93brhdVX9myR3a639elX9ZJKHJ6kk709y2hzqAwAAAABgP7S7qT/+IMlXkqS19trW2n9srR2XYdT1H8y2NAAAAAAA9le7C6+3t9Y+trqxtbYzyfaZVAQAAAAAwH5vd+H1bdY577abWQgAAAAAAKzYXXj9war65dWNVfXUJB+aTUkAAAAAAOzv1v3BxiTPSfK6qvrZ7AqrdyS5VZKfmGFdAAAAAADsx9YNr1trX0jy0Kp6VJL7jc1vbK39/cwrAwAAAABgv7W7kddJktbaO5K8Y8a1AAAAAABAkt3PeT0TVXVcVV1YVRdU1elVdZuqOrCq3lZVnxr/b51Y//iquqSqLq6qxy6iZgAAAAAA5mfu4XVVHZLk2Ul2tNbul+SAJEcl+Y0kZ7XWDk9y1ng6VXXEeP59kzwuyUur6oB51w0AAAAAwPwsZOR1hulKbltVW5LcLsnnkxyZ5NTx/FOTPH5cPjLJq1pr17bWLk1ySZIHzbdcAAAAAADmae7hdWvtc0lenOSzSa5M8uXW2luT3K21duW4zpVJ7jpe5JAkl09cxRVjGwAAAAAA+6hFTBuyNcNo6sOS3D3J7avq59a7yJS2tsZ1H1tVO6tq59VXX733xQIAAAAAsBCLmDbkh5Jc2lq7urX2r0lem+ShSb5QVQcnyfj/qnH9K5LcY+Lyh2aYZuQmWmsnt9Z2tNZ2bNu2bWZ3AAAAAACA2VpEeP3ZJA+uqttVVSV5dJKPJzkzydHjOkcnef24fGaSo6rq1lV1WJLDk5w755oBAAAAAJijLfO+wdbaB6rqNUk+nOS6JB9JcnKSOyQ5o6qemiHgfuK4/oVVdUaSi8b1n9lau37edQMAAAAAMD9zD6+TpLX2giQvWNV8bYZR2NPWPzHJibOuCwAAAACAPixi2hAAAAAAAFiX8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7mxZdAHsu7Yet3XRJUx1zUnXLLoEAAAAAGA3jLwGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7Cwmvq+ouVfWaqvpEVX28qh5SVQdW1duq6lPj/60T6x9fVZdU1cVV9dhF1AwAAAAAwPwsauT1/0ry5tbafZJ8T5KPJ/mNJGe11g5PctZ4OlV1RJKjktw3yeOSvLSqDlhI1QAAAAAAzMXcw+uqulOS70/yZ0nSWvtGa+0fkxyZ5NRxtVOTPH5cPjLJq1pr17bWLk1ySZIHzbNmAAAAAADmaxEjr789ydVJ/ryqPlJVL6+q2ye5W2vtyiQZ/991XP+QJJdPXP6KsQ0AAAAAgH3UIsLrLUm+L8nLWmvfm+SfM04Rsoaa0tamrlh1bFXtrKqdV1999d5XCgAAAADAQiwivL4iyRWttQ+Mp1+TIcz+QlUdnCTj/6sm1r/HxOUPTfL5aVfcWju5tbajtbZj27ZtMykeAAAAAIDZm3t43Vr7f0kur6rvGJseneSiJGcmOXpsOzrJ68flM5McVVW3rqrDkhye5Nw5lgwAAAAAwJxtWdDtPivJaVV1qySfTvILGYL0M6rqqUk+m+SJSdJau7CqzsgQcF+X5JmttesXUzYAAAAAAPOwkPC6tXZekh1Tznr0GuufmOTEWdYEAAAAAEA/FjHnNQAAAAAArEt4DQAAAABAd4TXAAAAAAB0Z1E/2Ahd23rc1kWXcBPXnHTNoksAAAAAgLkx8hoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAAAAgO4IrwEAAAAA6I7wGgAAAACA7givAQAAAADojvAaAAAAAIDuCK8BAAAAAOjOlkUXAGyercdtXXQJU11z0jWLLgEAAACAJWPkNQAAAAAA3RFeAwAAAADQHeE1AAAAAADdEV4DAAAAANAd4TUAAAAAAN0RXgMAAAAA0B3hNQAAAAAA3RFeAwAAAADQHeE1AAAAAADdEV4DAAAAANAd4TUAAAAAAN0RXgMAAAAA0B3hNQAAAAAA3RFeAwAAAADQHeE1AAAAAADdEV4DAAAAANAd4TUAAAAAAN0RXgMAAAAA0B3hNQAAAAAA3RFeAwAAAADQHeE1AAAAAADd2bLoAgCSZOtxWxddwk1cc9I1iy4BAAAAYL9l5DUAAAAAAN0RXgMAAAAA0B3hNQAAAAAA3RFeAwAAAADQHeE1AAAAAADdEV4DAAAAANAd4TUAAAAAAN0RXgMAAAAA0B3hNQAAAAAA3RFeAwAAAADQHeE1AAAAAADd2bLoAgCW2dbjti66hKmuOema3a7TY+0bqRsAAADYPxh5DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3FhZeV9UBVfWRqnrDePrAqnpbVX1q/L91Yt3jq+qSqrq4qh67qJoBAAAAAJiPRY68/tUkH584/RtJzmqtHZ7krPF0quqIJEcluW+SxyV5aVUdMOdaAQAAAACYo4WE11V1aJJ/n+TlE81HJjl1XD41yeMn2l/VWru2tXZpkkuSPGhOpQIAAAAAsACLGnn9B0mel+SbE213a61dmSTj/7uO7YckuXxivSvGNgAAAAAA9lFzD6+r6keTXNVa+9BGLzKlra1x3cdW1c6q2nn11Vff7BoBAAAAAFisLQu4zYcl+fGq+pEkt0lyp6r6yyRfqKqDW2tXVtXBSa4a178iyT0mLn9oks9Pu+LW2slJTk6SHTt2TA24AVhuW4/buvuVFuCak65ZdAkAAACwT5n7yOvW2vGttUNba9sz/BDj37fWfi7JmUmOHlc7Osnrx+UzkxxVVbeuqsOSHJ7k3DmXDQAAAADAHC1i5PVaXpjkjKp6apLPJnlikrTWLqyqM5JclOS6JM9srV2/uDIB4ObpcdS4EeMAAAD0aqHhdWvt7CRnj8v/kOTRa6x3YpIT51YYAAAAAAALNfdpQwAAAAAAYHeE1wAAAAAAdKenOa8BgA71OFd3Yr5uAACAfZ2R1wAAAAAAdEd4DQAAAABAd0wbAgDss3qc8sR0JwAAABsjvAYA6EyPoXsieAcAAObLtCEAAAAAAHTHyGsAADZNj6PGNzJivMe6E6PdAQDYvxl5DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3zHkNAABLrMf5us3VDQDAZhBeAwAAc9dj6J4I3gEAeiK8BgAA2AM9Bu9CdwBgX2TOawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALojvAYAAAAAoDvCawAAAAAAuiO8BgAAAACgO8JrAAAAAAC6I7wGAAAAAKA7wmsAAAAAALqzZdEFAAAAMHtbj9u66BKmuuaka3a7To+1L2vdycZqB4AeCK8BAACAG+kxeBe6A+x/TBsCAAAAAEB3hNcAAAAAAHTHtCEAAADAPqHH6U6S5Z0j3VQtwKIJrwEAAAC4WXoM3RPBO+wrhNcAAAAA7Hd6DN6F7nBj5rwGAAAAAKA7Rl4DAAAAwJLoccR4YtQ4s2HkNQAAAAAA3RFeAwAAAADQHdOGAAAAAAAz1+OUJxuZ7qTHupP9Y6oWI68BAAAAAOiO8BoAAAAAgO6YNgQAAAAAYB/U45QnezLdiZHXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd+YeXlfVParqHVX18aq6sKp+dWw/sKreVlWfGv9vnbjM8VV1SVVdXFWPnXfNAAAAAADM1yJGXl+X5Ndaa9+Z5MFJnllVRyT5jSRntdYOT3LWeDrjeUcluW+SxyV5aVUdsIC6AQAAAACYk7mH1621K1trHx6Xv5Lk40kOSXJkklPH1U5N8vhx+cgkr2qtXdtauzTJJUkeNNeiAQAAAACYq4XOeV1V25N8b5IPJLlba+3KZAi4k9x1XO2QJJdPXOyKsQ0AAAAAgH3UwsLrqrpDkr9O8pzW2j+tt+qUtrbGdR5bVTuraufVV1+9GWUCAAAAALAACwmvq+qWGYLr01prrx2bv1BVB4/nH5zkqrH9iiT3mLj4oUk+P+16W2snt9Z2tNZ2bNu2bTbFAwAAAAAwc3MPr6uqkvxZko+31n5/4qwzkxw9Lh+d5PUT7UdV1a2r6rAkhyc5d171AgAAAAAwf1sWcJsPS/LzSc6vqvPGtv+c5IVJzqiqpyb5bJInJklr7cKqOiPJRUmuS/LM1tr1c68aAAAAAIC5mXt43Vp7T6bPY50kj17jMicmOXFmRQEAAAAA0JWF/WAjAAAAAACsRXgNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdEd4DQAAAABAd5YmvK6qx1XVxVV1SVX9xqLrAQAAAABgdpYivK6qA5L87yQ/nOSIJE+uqiMWWxUAAAAAALOyFOF1kgcluaS19unW2jeSvCrJkQuuCQAAAACAGVmW8PqQJJdPnL5ibAMAAAAAYB9UrbVF17BbVfXEJI9trf3SePrnkzyotfasVesdm+TY8eR3JLl4RiUdlOSLM7ruWVrWupPlrX1Z606Wt/ZlrTtZ3tqXte5keWtf1rqT5a19WetOlrf2Za07Wd7al7XuZHlrX9a6k+WtfVnrTpa39mWtO1ne2pe17mR5a1/WupPlrX1Z606Wt/ZZ1/1trbVtqxu3zPAGN9MVSe4xcfrQJJ9fvVJr7eQkJ8+6mKra2VrbMevb2WzLWneyvLUva93J8ta+rHUny1v7stadLG/ty1p3sry1L2vdyfLWvqx1J8tb+7LWnSxv7ctad7K8tS9r3cny1r6sdSfLW/uy1p0sb+3LWneyvLUva93J8ta+qLqXZdqQDyY5vKoOq6pbJTkqyZkLrgkAAAAAgBlZipHXrbXrqupXkrwlyQFJXtFau3DBZQEAAAAAMCNLEV4nSWvt75L83aLrGM18apIZWda6k+WtfVnrTpa39mWtO1ne2pe17mR5a1/WupPlrX1Z606Wt/ZlrTtZ3tqXte5keWtf1rqT5a19WetOlrf2Za07Wd7al7XuZHlrX9a6k+WtfVnrTpa39oXUvRQ/2AgAAAAAwP5lWea8BgAAAABgPyK8XqWqvrroGuatqp5WVU8Zl4+pqrsvuqZpquruVfWacfmRVfWGOdzm9qq6YBOu55iq+qNx+fFVdcTEeWdX1U1+rbWqdlTVSzbhtu9SVc8YlzftcZu8T4tQVc+uqo9X1WmLquHmWOljJrfn8fTpVfWxqjpucdVNV1VPHB/rd4ynb6i1qn67qn5o0TXeXFX1vkXXsL+rqsuq6qC9XWcexn3Czyzwtvd6f7Sb6/+ZidObsg9adf0zq3+8jZm/nte6H2vtyxdprf3kZj63GzkW2N3+uqruX1U/MnH6hKp67mbUt5u69rtj/lmpqlOq6glT2udyvL4789qmNkNVvXzlfcIi930rtz35PmLOt/+cqrrdBtbb7et4nX578rHeUH+wyH5js977TB67T+67qurvquoum1DqptQ25byp/cwymdzmejGL45e97fuX9X3+zTG5XVfVI6rqwqo6r6oOmcwKlsnq47rNsjRzXjM7rbU/njh5TJILknx+MdWsrbX2+SRLvcMaPT7JG5JctN5KrbWdSXZuwu3dJckzkrx0E66rJ89I8sOttUsXXcjNMbk9V9W3Jnloa+3bFlvVmp6a5BmttXdstNaqOqC1dv18yrv5WmsPXXQNi1ZVlWEasW8uupYlsD3JzyT5PwuuYxa2Z+K+beI+aG68nm/iJvvJqtqygOd2d/vr+yfZkU36bZtl2f/ANK21X9rM69uEffxdspj3Ec9J8pdJvjarG9jsx3oONuW9T2vt+Wu0b3rYtKfWqq2qDph3LbOwhNvcomx4Wx+Pa66bQ03z8LNJXtxa+/Px9LJmX/fPJh7XrTDyeg01eFFVXVBV51fVk8b2v1o1OuSUqvqpqjpgXP+D44jE/zCnOrdX1Seq6tTxdl9TVberqkdX1UfG2l9RVbce139hVV00rvvise2Eqnru+InPjiSnjZ/23HZGNd++qt5YVR8dH98njZ/u//eqen9V7ayq76uqt1TV/62qp03c12mfmt9+vI8fHO/zkZtc8gFV9afjp2BvrarbVtW9qurNVfWhqnp3Vd1nrOXHquoDYx1vr6q7rar1oUl+PMmLxsf4XuNZT6yqc6vqk1X1iHHdzRqt8sIk96qq85K8KMkdxu3kE1V12nhQm6p6/vgYXlBVJ0+0n11Vv7u6vlX369+Pz91cRodU1R8n+fYkZ1bVl2tiNM1Y//bx7+Orn7t51LcRq7bntya567hNPGKt7WtOdf3NeLsXVtWxVfX8JA9P8sdV9aIptU5+WnzZuB29J8M2vdvX9aLVnEbQrNPvHTSev6Oqzh6XT6ihT3/ruM5PVtXv1dCfv7mqbjmut6HHt6p+vXbtm35rbFt5fbw0yYeT3GMP78/Kvufl4/05rap+qKreW1WfqqoHVdWB4/b0sao6p6q+e7zst4z37SNV9SdJauJ6f27sa86rqj+pOb1ZqaqnjHV+tKr+olaN7pnYTl6Y5BFjfcfV/Pf9W+qm+/sHVNU7x9ftW6rq4LHmXx7r+mhV/XWNI9j24L7NYsTktP3penX+cQ194Cer6kfH9mOq6vXja+HiqnrBlPsyazd5HibPrKonj6/XC6rqdyfanzrel7PHx2Fm316qm+4nT66qtyZ55eRzW2scQ42P82vHx/lTVfV7E9f9C+P9eGeSh+1BHf+pqt433s77quo7qupWSX47yZPGbe9J40WPGB+nT1fVsyeub2ofUVVfrWHE3geSPKSmH+8eVkN/+cGq+p2J67xDVZ1VVR8en7eVx+B3qupXJ9Y7cbKWPXw+NtJnzvp4dk9r/q9jzW+r4RtXz61hNNU54+P6uqraOuVyjxsv954kP7mA0lfq+M2xj3h7ku8Y227S31TVHavq0tq1b71TDfvXW86hxmnHBjcZBVnDcfgzJk6fUFW/Ni7v9T6+Vh37rTr7hvcRNezvXlpVPz5e7nVV9Ypx+alV9d/Wur7x/JMmbvOXq+r313gMnp3k7kneUbu+9Te1Xx3PO3G8/Dm16r3XhGn7zxs91tOuZ5H9xqr7uG5fOq5zzPjY/+24Tf9KVf3Hcb1zqurAcb0bHQdM3MYNx6XzUNP7mDXfV8yrron6NtJvn1DT34feZLsez79hm6uhr/zwuM5Ze1DX81a2qao6qar+flx+dFX9ZVU9ZtxmP1xVr66qO4znTz1enLjeW4yvkf9Waxzf1nD8cHZNzxI2pe9fta3/Wk1/L3FC3fi4ZlsNffoHx7+Hjeu9vnbNMPAfak4juac9/xt4/H8pyU8nef74uG6vGX9jcY3ab9h3TrwuJ7fbg6rqsnH5NlX15zX0gR+pqkfV2sd1e6+15m/iL8lXx/8/leRtSQ5Icrckn01ycJKfSHLquM6tklye5LZJjk3yX8b2W2cY0XLYHOrdnqQledh4+hVJ/stY173Htldm+PT6wCQXJzf8UOddxv8nJHnuuHx2kh0zrvmnkvzpxOk7J7ksydPH0ycl+ViSOybZluSqift6wbj8yCRvGJf/e5KfW7lPST6Z5Pab+Phel+T+4+kzkvxckrOSHD62/dskfz8ub514fH8pyf8cl49J8kfj8ilJnjBxG2dPrPcjSd6++j5uwn2YfNy+nOTQDB9evT/Jw8fzDpy4zF8k+bHd1HdMkj/K8Jp4d5Kts97eV92vy5IcNLn9ju0XjPd56nM3zxrXqHulj5l8Xm5YHk9P3b7mVN+B4//bjo/lt2SiX5hS6w3b8/icPG/Vc7Tu63rRfyvPxxxuZ61+76Dx9I4kZ4/LJyR5T5JbJvmeDKOOfng873VJHr/RxzfJYzL8InSNr/k3JPn+8Xn8ZpIH38z7s/L6+q7xej+UYf9TSY5M8jdJ/jDJC8b1fzDJeePyS5I8f1z+9xn2YQcl+c4kf5vkluN5L03ylIn7etCMnpv7Ztg3rjwXB+am/fTK6/aRmeiXM8d9f6bv7389yfuSbBvbnpTkFePyt0xc9r8leda4vNH7dqPTm1T/tP3penW+edy+Dk9yRZLbZNj3XJmhb1rpp3ZM3pdZ/q3xPDw3Yz+ZIXD5bIbX4ZYkf5/hG1d3H7fjAzO8tt+d8bhghrVell37yQ8lue3q5zZrHEONj/OnM/RVt0nymQwB2MET9+9WSd67u/sxUcedkmwZ234oyV+Py8dMXsdY7/syvKYOSvIP42O2Xh/Rkvz0xGt42vHumRPrPzO7tv0tSe40Lh+U5JIMfdn2JB8e22+R5P9mYnu9mdv/en3mzI5nb0a9O5Kcl+E1dsckn8qwnX8syQ+M6/x2kj+YeL0+YdxWLs/wmq0Mr/NN60f2oP4HJDk/ye3G7e6Ssf61+ps/z65967EZj3vnUOe0Y4Ozs6tPW3ntfG+Sd06sd1GSe2aT9vGZfuy3ctvbc+PjvqOSvGhcPjfJOROP4WPXub7bj6+hldfv+8bXw00eg8n7Pi5P7VfH81p2vWf5vYz75CmvvzX77fWuJwvsN6bcj5XnZL2+9JLsOhb8cpKnjeedlOQ5k6/XcXnyMbjhMZ/Dtr9WHzNZ22W58fuKG86bU43bs/t++4RMfx+61nZ99njft2XoKw+bfM1ssK4HJ3n1uPzuDK/DWyZ5QZL/lORdGfcd4+nnj+evdbx49nidpyf5zbFt6vFt1sgSssl9f3Zt62u9lzghNz6u+T/ZlWncM8nHx+W7ZXhNPCLDPnXDj/NebjvT+va1Hv/JbX5yeXsm+t451b3WvvPs7OonDkpy2bj8a0n+fFy+T4Z+euVYfdOPcU0bsraHJzm9DV87/EINo0semORNSV5Sw0jmxyV5V2vtX6rqMUm+u3Z9innnDC/eS+dQ6+WttfeOy3+Z5L8mubS19smx7dQMO9s/SvL1JC+vqjdmOMBZhPOTvLiGT83f0Fp79/iB3ZkT59+htfaVJF+pqq/X+vNvPSbJj0986nmbjJ3WJtV7aWvtvHH5Qxk6kocmefVYdzJ06snQkf/V+EnarbLx5/+1q65/ls5trV2RJDWMxt6eISh7VFU9L0NndWCSCzO8SVyvvkdl2AE/prX2TzOu++aY9tx1bfx0fK3tax6eXVU/MS7fI0M/tif+atXpdV/XrbV/vPmlLpW1+r21vKm19q9VdX6GD1HfPHE92yfW212/+Zjx7yPjenfI8Jx+NslnWmvn7MV9urS1dn6SVNWFSc5qrbWx5u1Jvi3DwVtaa39fw4jrO2d4Y/2TY/sbq+qa8foeneGg6YPjY3PbJFftRX0b9YNJXtNa++JY05d289xMmve+f/X+/j8nuV+St401H5Ah2E2S+9UwCu4uGZ73t8yopj0xrU9er84z2vBV909V1aczHBgnydtaa/+QJFX12gzHbPOcBmP18zA5qu6BGT6Iunqs77QM23wyhE9fGttfneTec6o3Sc5srf3LlPa1jqGS4TX95SSpqosyvKYPyo3v319l4/fjzklOrarDMwRF641sfWNr7dok11bVVRnegK7XR1yf5K/H5X/K9OPdh2XskzJ8SL8yerOS/Peq+v4Mgd8hSe7WWrusqv6hqr53vP2PrGx3N9Pu+sxDM9vj2T3x8CSvX9lmqupvM4SPd2mtvXNc59Qkr151uftkuJ+fGi/3lxlCkHl7RJLXtda+Ntaxsq9cq795eZLnZQijfiHJL8+pzg0dG7TWPlJVd63hd4m2JbmmtfbZGkZfbsY+fk+O/d6d5Dk1zNt7UZKt4/ueh2RXX3iT62utnVPDCNEfraqPZwixz6+qa1c/BlNuc61+9W+SfCO7XuMfSvLv1qh7vX4761zPovuNadbrS98xcSz45ex6L3d+ku/e5Dr2xrQ+ZprV7yvmbXf99nlrXO4mr+1V5z84Q450aTIce+5BTR9K8oCqumOSazN8u2JHhn7vzCRHJHnv2JfcKkPA/B1Z+3gxSf4kwzHXiePptY5vv5HpWcJXM5u+/+GZ/l4iufFxzQ9l+MbWyuXuVFV3bK19oYZvEb8jyU/s4eO8N270/Ce5Jus//r1Ya9+5lodn+IAhrbVPVNVnMsNjW+H12qa+c22tfb2Gr3Y/NsMnJqdPrP+s1toi3hy2Da3U2nVV9aAMB/9HJfmVDG/a56q19smqekCGUbz/o4aveyRD55sMBwDXTlzkm1l/W60kP9Vau3jTi71xXcnw5uhuSf6xtXb/Kev+YZLfb62dWVWPzPCp4J7cxvWZ/ety9f3ZUlW3yTCCaUdr7fKqOiHDm6bd1ffpDF/ruXcWNzfqdbnxFEjT6k6G2ruZNmQdt8ja29dMjdvsDyV5SGvta2Nfd5v1LjPFP686fXNf1/uUNfq9yW139eN87Xi5b1bVv7bxI+3c9HHb3eNbSf5Ha+1PJq+8qrbnps/Vnlp9e5O1bMlw/1Zrq/7fqKwM32w6fi/r2lM1pZ4bnpsajjJvtc5l57nvX13nV5Jc2Fp7yJR1T8kwMu2jVXVMhtEyycbv2yxM65NPyfQ6k5ve37W2nw0dB22i9W5/rU8+NvyJyIys9XqfegxVVf82U44XxuWb+3j/ToZg5SfGPujsddaddtvr9RFfHwec7O54d1rtP5shEHzA+KHhZdnVJ788wwiib80w4m5v7K7PvD6zPZ7dE3uzvc779biWaXWckin9TWvtvTV8RfsHkhzQWpvLV7XXeU80zWsyjG7/1iSvGtv2eh+/p8d+rbXP1TBdzOMyjPA8MMNX3b/aWvvKbq7v5Rk+dP1EhpHaUx+D1tpvry5znbsweYy03vuo3e031rueRfYb06zXl+6un+nFRvuYvT1W3VsbOda9yfvQDWzX0449N2Rie/uFDKN5P5ZhQNm9MgyeeFtr7cmTl6mq78rax4sZr+dRVfU/W2tfzxrHt+Pre7OPDdYzbTtZuZ3JbeMWGfqcaR/Sf1eGb3DdfZNrW9Pq5z/DjA7rPf49mfY8rvWeda7Htua8Xtu7MszTckBVbcvw6e6543mvytBZPCK7PrF/S5Kn16750u5dVbefU633rKqVF8KTk7w9yfaq+jdj288neWcNIzrv3Fr7uwzTiNx/ynV9JcNXd2ZmHDXwtdbaXyZ5cZLv28urfEuSZ41vwjN+0j1L/5Tk0qp64nh7VVXfM5535ySfG5ePXuPyM3+Mb8btrXRCXxy3k5vMg7aGz2QYQfnKqrrvzaxvb12WcRuqqu/L8JWmpdWGEexrbV+zducMo3m+VsM82w+e0+3u89bo9y7LMIow2TWqZ7O9Jckv1q757g6pqrvO6LZWe1eGN3crB7tfHLfvyfYfzjDdUjJMl/OElfpqmDP72+ZQ51lJfrqqvmXldnPj5+bI7BrVtLo/nfe+f/X+/pwk21baquqWE33xHZNcOdb2sxPXcVk2dt/mZa06k2Hu/FvU8PsQ355hKogk+Xfj9nHbDFNyvDfztfp5eM/EeR9I8gM1zAl4wHj+OzMcQ/5AVW2tqi2Z3Wt+T+3pMdQHkjxyHP10y+zZPKSTx0jHTLRvdNvbUB+xzvHuezOE2cmNt7U7Z5hq6V+r6lEZRpiveF2GkO6Bmf23F+Z9PLue9yT5sRrms7xDhime/jnJNbXrt09+PsO2PekTSQ6rXb/p8uQsxruS/EQN8+rfMcmPje3r9TevzDAo6c/nVeQevid6VYbt9wkZguxkc/bxuzv2m/b6fH+G19a7MozEfu74f93ra619IMNI7J/JOABsncdg8nbX6lf3xHr99np67DfW6kuXybQ+ZhldlinvQzfw2n5/hm16Zf0D9/B235XhdbfyGnxahlHg5yR52EoOVMPc7vfOcPy01vFikvxZhh/Xe/V4jLKnx7ez6vvXei+x2lszfFCdcd37j/8flOSHM0y99NyVx3vWpjz//zbrP/69WGvfeVl2vXd4wqr1V56fe2f4ttjFmdF7CuH12l6X4VOsj2aYV+t5rbX/N5731gxh9ttba98Y216e4atTH65hYvU/yfw+3fx4kqOr6mMZPv0+KUO4/uoavtLyzSR/nGEDesO43juTHDfluk7J8MNs59Xsftzuu5KcW8PXTH4zw5xze+N3Mrz5/tj42P/ObtbfDD+b5KlV9dEM02scObafkOFxf3eSL65x2Vcl+fUaJrW/1xrrbJrxq2rvHR+bF62xzj8m+dMMX3H5myQf3IPrvzjD4/HqedyfKf46yYHj9vT0DPNZLbu1tq9Ze3OGkfgfy/A62pspJbixaf3ebyX5X2N/cf0sbrS19tYM88C9f9wfvCbzCyhPSLJj3J5emF0f6P1Wku+vqg9n+FriZ8daL8rwmw1vHS/ztgzz685Ua+3CJCdm+JD3o0l+P0N/+ANVdW6GA86V0R0fS3JdDT/Aclzmv+9fvb//wwwHkb871n5ehmmHkmEKsQ9keBw/MXEdG71v87JWnclwAPzODFO2PW0cDZQMb3r/IsP9/evW2ry/+bP6eXjZyhmttSuTHJ/hK6ofzTD36etba5/LMKfxBzIMMrgow7yRi7ZHx1Dj/Tshwxvvt2f4uvJG/V6GEWjvzfCV2RXvyPB133V/2GcP+oi1jnd/Nckzq+qDGcKfFadl6Kt2Ztj/3rAdjsf578jwdeqZ9NMTFnE8O1Vr7YMZvn7+0QxTx+3MsL0eneFHxz+W4UOB3151ua9n+Kr4G2v40a7PzLHsyTo+nGG6gfMyHCeuBKvr9TenZfgw9fTMz4bfE437qjsm+dz4Otysffy6x36T7yNq+OHuZHg8t7TWLsnQBxyYXY/x7o4lz0jy3tbaypRhaz0GJyd5U1W9Y61+dQ/v55r99m702G+s1ZcujXX6mGWz1vvQdV/bbZgC59gkrx2P3/Z0epR3Z9j/vb+19oUMU2W9e7zeY5KcPm7r5yS5z7hNrnW8uFLT72d4Pf9F9vD4doZ9/wmZ/l5itWevrFfDNGdPq2Ga3z9N8outtc9nmJ/5FSsfEM/Y6uf/+dnN49+DdfadL87wYcb7Mkwft+KlGX6M/fzxcse0Ycq3DR3X7amVHzJhSdXwVaE3tNbut+haAAA2Q1WdkuH45jWr2o/JMMXVr0y7XM+q6g6tta+Oo5pel+HHel636LpYW1XdIsOb+Se2cS7P/cXE9nq7DKOrjh3f2O6Tapjb9cjW2s8vupZ9WVW9IclJrbWzFl3LrOzP/cae2N/6GFg2NUwl+9XW2osXXUti5DUAAMzDCeMonAsyzEv5NwuthnXV8IN0l2T4ga79MYA6edxeP5zhGw77bKhUVX+YYVTfwka77+uq6i5V9ckk/7KPB9f7e7+xJ/abPgbYe0ZeAwAAAADQHSOvAQAAAADojvAaAAAAAIDuCK8BAAAAAOiO8BoAAGakqr666vQxVfVHN+N6HllVb1jjvOdU1e1ubo0AANAr4TUAACy35yQRXgMAsM8RXgMAwAJU1Y9V1Qeq6iNV9faqutvY/gNVdd7495GquuN4kTtU1Wuq6hNVdVoNnp3k7kneUVXvGC//sqraWVUXVtVvTdzej4yXfU9VvWStkdwAANCLaq0tugYAANgnVdX1Sc6faDowyZmttV+pqq1J/rG11qrql5J8Z2vt16rqb5O8sLX23qq6Q5KvJ3l4ktcnuW+Szyd5b5Jfb629p6ouS7KjtfbF8TYPbK19qaoOSHJWkmcn+WSSTyX5/tbapVV1epI7ttZ+dA4PAwAA3CxbFl0AAADsw/6ltXb/lRNVdUySHePJQ5P8VVUdnORWSS4d29+b5Per6rQkr22tXVFVSXJua+2K8XrOS7I9yXum3OZPV9WxGY71D05yRIZvXH66tbZyG6cnOXZz7iIAAMyGaUMAAGAx/jDJH7XWvivJf0hymyRprb0wyS8luW2Sc6rqPuP6105c9vpMGYhSVYcleW6SR7fWvjvJG8frrVndCQAAmBXhNQAALMadk3xuXD56pbGq7tVaO7+19rtJdia5z7QLT/hKkpV5se+U5J+TfHmcQ/uHx/ZPJPn2qto+nn7S3pcPAACzZdoQAABYjBOSvLqqPpfknCSHje3PqapHZRhdfVGSNyV5yDrXc3KSN1XVla21R1XVR5JcmOTTGaYgSWvtX6rqGUneXFVfTHLuLO4QAABsJj/YCAAA+4GqukNr7as1TKD9v5N8qrV20qLrAgCAtZg2BAAA9g+/PP7Q44UZpiz5k8WWAwAA6zPyGgAAAACA7hh5DQAAAABAd4TXAAAAAAB0R3gNAAAAAEB3hNcAAAAAAHRHeA0AAAAAQHeE1wAAAAAAdOf/A2/+SBK1jby9AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1800x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# non-racist tweets\n",
    "\n",
    "non_racist_tweets = nltk.FreqDist(hash_regular)\n",
    "# non_racist_tweets.keys() are hashtags \n",
    "# non_racist_tweets.values() are frequency of hashtags\n",
    "df1 = pd.DataFrame({'Hashtag': list(non_racist_tweets.keys()), 'Count': list(non_racist_tweets.values())})\n",
    "# creat a data fram of two columns representing each hashtag with its freq \n",
    "\n",
    "# selecting top 30 most frequent hashtags\n",
    "df1 = df1.nlargest(columns=\"Count\",n=30) # number of columns in figure\n",
    "plt.figure(figsize=(25,10)) # size of figure\n",
    "ax = sns.barplot(data=df1, x=\"Hashtag\", y=\"Count\",color=\"green\") # determine a-axis and y-axis\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "8367bd35",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1800x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#racist tweets\n",
    "\n",
    "racist_tweets = nltk.FreqDist(hash_racist)\n",
    "# racist_tweets.keys() are hashtags \n",
    "# racist_tweets.values() are frequency of hashtags\n",
    "df2 = pd.DataFrame({'Hashtag': list(racist_tweets.keys()),'Count': list(racist_tweets.values())}) #count number of occurrence of particular word\n",
    "\n",
    "#selecting top 30 frequent  hashtas\n",
    "df2 = df2.nlargest(columns = \"Count\",n=30) # number of columns in figure\n",
    "plt.figure(figsize=(25,10)) # size of figure\n",
    "ax = sns.barplot(data=df2, x=\"Hashtag\",y=\"Count\",color=\"red\") # determine a-axis and y-axis\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "id": "b0f6f6f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# extract features from data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "id": "922c62e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer \n",
    "import gensim "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "id": "796b8ec4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(49159, 1000)"
      ]
     },
     "execution_count": 183,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Bag-of-words\n",
    "\n",
    "#Each row in matrix M contains the frequency of tokens(words) in the document D(i)\n",
    "# ignore terms that appear in more than 90% of documents and ignore terms that appear in less than 2 documents\n",
    "bow_vectorizer = CountVectorizer(max_df=0.90 ,min_df=2 , max_features=1000,stop_words='english') \n",
    "# build matrix and vocabloary\n",
    "\n",
    "bow = bow_vectorizer.fit_transform(combine['raw_tweet']) # tokenize and build vocabulary\n",
    "# for more information https://www.geeksforgeeks.org/python-ways-to-sum-list-of-lists-and-return-sum-list/\n",
    "# https://stackoverflow.com/questions/27697766/understanding-min-df-and-max-df-in-scikit-countvectorizer\n",
    "bow.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e713a48",
   "metadata": {},
   "source": [
    "# replace null values with 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "65567fee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>label</th>\n",
       "      <th>tweet</th>\n",
       "      <th>raw_tweet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run</td>\n",
       "      <td>father dysfunct selfish drag kid dysfunct #run</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked</td>\n",
       "      <td>thank #lyft credit use caus offer wheelchair van pdx #disapoint #getthank</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>bihday your majesty</td>\n",
       "      <td>bihday majesti</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>#model   i love u take with u all the time in urð±!!! ðððð",
       "ð¦ð¦ð¦</td>\n",
       "      <td>#model love u take u time ur</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>factsguide: society now    #motivation</td>\n",
       "      <td>factsguid societi #motiv</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>[2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo</td>\n",
       "      <td>huge fan fare big talk leav chao pay disput get #allshowandnogo</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user camping tomorrow @user @user @user @user @user @user @user dannyâ¦</td>\n",
       "      <td>camp tomorrow danni</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl</td>\n",
       "      <td>next school year year exam think #school #exam #hate #imagin #actorslif #revolutionschool #girl</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "      <td>we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦</td>\n",
       "      <td>love land #allin #cav #champion #cleveland #clevelandcavali</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>@user @user welcome here !  i'm   it's so #gr8 !</td>\n",
       "      <td>welcom #gr</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  label  \\\n",
       "0   1    0.0   \n",
       "1   2    0.0   \n",
       "2   3    0.0   \n",
       "3   4    0.0   \n",
       "4   5    0.0   \n",
       "5   6    0.0   \n",
       "6   7    0.0   \n",
       "7   8    0.0   \n",
       "8   9    0.0   \n",
       "9  10    0.0   \n",
       "\n",
       "                                                                                                                                             tweet  \\\n",
       "0                                            @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run   \n",
       "1                       @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked   \n",
       "2                                                                                                                              bihday your majesty   \n",
       "3                                                           #model   i love u take with u all the time in urð±!!! ðððð\n",
       "ð¦ð¦ð¦     \n",
       "4                                                                                                           factsguide: society now    #motivation   \n",
       "5                             [2/2] huge fan fare and big talking before they leave. chaos and pay disputes when they get there. #allshowandnogo     \n",
       "6                                                                        @user camping tomorrow @user @user @user @user @user @user @user dannyâ¦   \n",
       "7  the next school year is the year for exams.ð¯ can't think about that ð­ #school #exams   #hate #imagine #actorslife #revolutionschool #girl   \n",
       "8                                                          we won!!! love the land!!! #allin #cavs #champions #cleveland #clevelandcavaliers  â¦    \n",
       "9                                                                                                @user @user welcome here !  i'm   it's so #gr8 !    \n",
       "\n",
       "                                                                                         raw_tweet  \n",
       "0                                                   father dysfunct selfish drag kid dysfunct #run  \n",
       "1                        thank #lyft credit use caus offer wheelchair van pdx #disapoint #getthank  \n",
       "2                                                                                   bihday majesti  \n",
       "3                                                                     #model love u take u time ur  \n",
       "4                                                                         factsguid societi #motiv  \n",
       "5                                  huge fan fare big talk leav chao pay disput get #allshowandnogo  \n",
       "6                                                                              camp tomorrow danni  \n",
       "7  next school year year exam think #school #exam #hate #imagin #actorslif #revolutionschool #girl  \n",
       "8                                      love land #allin #cav #champion #cleveland #clevelandcavali  \n",
       "9                                                                                       welcom #gr  "
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "combine=combine.fillna(0) #replace all null values by 0\n",
    "combine.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "e86535bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(bow, combine['label'],\n",
    "                                                    test_size=0.2, random_state=69)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "id": "ebcc9cd3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train_shape :  (39327, 1000)\n",
      "X_test_shape :  (9832, 1000)\n",
      "y_train_shape :  (39327,)\n",
      "y_test_shape :  (9832,)\n"
     ]
    }
   ],
   "source": [
    "print(\"X_train_shape : \",X_train.shape)\n",
    "print(\"X_test_shape : \",X_test.shape)\n",
    "print(\"y_train_shape : \",y_train.shape)\n",
    "print(\"y_test_shape : \",y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "10479ac1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier\n",
    "\n",
    "model_naive = MultinomialNB().fit(X_train, y_train) \n",
    "predicted_naive = model_naive.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "43b91e5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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SZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSZIkSZIkSR0stpAkSZIkSZIkSZIkSZIkSZIkSepgsYUkSZIkSZIkSZIkSZIkSZIkSVIHiy0kSZIkSZIkSZIkSZIkSZIkSZI6WGwhSZIkSZIkSZIkSZIkSZIkSZLUwWILSZIkSZIkSZIkSZIkSZIkSZKkDhZbSJIkSZIkSZIkSZIkSZIkSZIkdbDYQpIkSZIkSZIkSZIkSZIkSZIkqYPFFpIkSZIkSZIkSZIkSdL/Z+++w/2ez/+BP9/Ze8sgZkQINWtvpRQ1OmirtFYVRc1+W7to7b07FK2WqqI2Va299ybTSAgZZCc+vz/Ir296zufsk4jH47rOJTn3fV73/UldruZzzvP9AgAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAAAAAAAAAAAAAAgBJhCwAA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KpUZjdilT5IBSZZMMjTJcklWT7JKki517PPZwMVSSW4rimLtSqUyuaG7AABAQwwfVv1pXknywCNPZMnFm/8CtvreMPDh1GmZPWdO2rer+6+5dYU3unbtUrXeXLp17ZJp02u+aSJJJk/5sFnmTJ7yQU4++5Iaa+3ats2xPzuoWb8huLC4/uY788zzL+X8U47LEoOrf5MVAAAAAGg5SyyxWE466edVe2bNmpXTTr+wSXMGDexftf7WW29nueWG5M47rsmiiw5s0qzatGnTJocdul+GDl0m3/vevpk1a1aLzOndu1fOPOOXNdZmz56dffc7IpXKZ58XCgAAXzx+moKFVlEUyyW5IrUHLT57g8WsJDck+VGSpSuVymKVSmX7SqVyTKVSubxSqdxfqVRGNSZokSSVSuX9SqXyYqVSua1SqZxXqVQOqFQq6yfpmWT9JMcmeTb/e5vFp15W6XNDk/ypMbsAAEBD9F+kb51BimtuuKVFZl9zff3PnfJB/cIJdfV169I6YYu6Qh31udGjPk4999K8X8vT3HbdeccMW3bpZpmzMBoxamx22efgvDZi9PxeBQAAAAC+kIqiyO9+e3Z69Ohete/c836bEU14H69jx47p1atn1Z42bdrktlv/0mJBi7Ltt9sqV199aYs9KOf0047NgAGL1Fg759zf5NlnX2yRuQAA8HnjZgsWZhcn6ZbagxbzPjciyTlJLq9UKlNab71PFqlU5iZ58JOPE4qiGJ7koCS7JumU/91/XuCiSPK1oih2qVQqQhcAALSoTdZfO5f/5bpa66++PiqPPvlM1lxt5Wab+cjjT+fVEaPq3T9rZv2e8FXXk8A6depY75lN0blTp6r1mc3wxLIHH30yN9x6V421RQf2z3577tLkGfPb0GWWyvBhy2bokKUydJmlMnDAIunetWu6deua9u3bZfKUDzJ58pS8N3Fynnn+pTz+1LN58tkX8uHUafU6f+KkKdnrpz/PlRedkcUXG9TCrwYAAAAAKDv22MOyySbrVe0ZM+bNnHTS2U2a06tXjzp7zj/v11l88dpvwZ0y5YPcfsc9uemmO/L88y9n3Lh38v77k9KnT68MHNg/K644LNtu+9Vs+dVN6gyPJMm222yRU085JocdflxDXkqdNttsw+y220411kaNGptf/vKMZp0HAACfZ8IWLJSKotgpySb536BC+feT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GFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGDi86ADzVFV/u+gMC9Td/bBFhwAAAAAAAAAAAAAAgP1upcoWSR6SpBcdYgEqq/m9AQAAAAAAAAAAAABgZqtWthipRQeYIyULAAAAAAAAAAAAAACYwaqWLRQQAAAAAAAAAAAAAACAiVa1bGGyBQAAAAAAAAAAAAAAMNGqli0UEAAAAAAAAAAAAAAAgIlWtWyxSpMtAAAAAAAAAAAAAACAGaxa2eJFMdUCAAAAAAAAAAAAAADYxEqVLbr7IYvOAAAAAAAAAAAAAAAA7G+HFh0AAAAAAAAAAAAAAABgP1G2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGDi86ACroqrOS/KZSe6V5NIkt0vyiUnOTXJGktMGb+/u9rMBAAAAAAAAAAAAAIAF8Af9e6iqbpXka5N8eZL75sRJIrVLz7k4yR2m3H5Hd1+5G88BAAAAAAAAAAAAAIBVoGyxB6rqk5L8jySPS3LqaHnCW3vSx7fxyPOSvHDKfi9N8qBt7AkAAAAAAAAAAAAAACtpfNICO1BVh6vqR5NcnuQbktwsa+WJyloRYvzY8PHtPre7X5vk/wyeNTw+o6qmTb0AAAAAAAAAAAAAAADGKFvskvVpFi9P8qQkp+XEgkUyuQwxOnbq59ZfJ5U5HrsL+wMAAAAAAAAAAAAAwEpQttgFVXXfJK9Kcs9sLFkkGwsVJ5tusW3d/Q9J3ji+vP5cZQsAAAAAAAAAAAAAANgiZYsdWi9avCDJrXO8UJFMLlgM13drosXQ7wz2HO79yVV1j11+FgAAAAAAAAAAAAAAHEjKFjtQVRcmeU6S83K8UDEsUYwXLG5M8rIkf5jkl5M8d+x9O/UHg73G93z4Lj0DAAAAAAAAAAAAAAAONGWLnXlWkvOzsVSRbCxeHEvyR0kekeTm3f3A7v6q7v72JM/ezTDd/e9JXpnJEzOULQAAAAAAAAAAAAAAYAsOLzrAsqqqb0rymZlctBhd/3WS7+juK+cY7blJ7je4HpU+LquqU7r7pjlmAQAAAAAAAAAAAACApWOyxTZU1elJfiwbixXDaRad5Ie7+/PnXLRIkhcNzocTLs5KcumcswAAAAAAAAAAAAAAwNJRttier09y/vr5sNAwKlp8d3f/1NxTrXllkmPr5z127y5zzgIAAAAAAAAAAAAAAEtH2WJ7vm7sejjR4pe7+xfnnmgUpPvDSd4+5bayBQAAAAAAAAAAAAAAnISyxYyq6vZJPj0bCxYj1yT5wUXkGvOWbJy4MaJsAQAAAAAAAAAAAAAAJ6FsMbuHTFgblS5+cn2yxKJdNWX94nmGAAAAAAAAAAAAAACAZaRsMbv/MjjvsfNnzDnLNO8Zux5N4ThvAVkAAAAAAAAAAAAAAGCpKFvM7k5j17X++uruvnbeYaZ4/5T1c+YZAgAAAAAAAAAAAAAAlpGyxewuzsaJFlm/fsX8o0z1sSnryhYAAAAAAAAAAAAAAHASyhazO2/K+nvnmmJz036uZ801BQAAAAAAAAAAAAAALCFli9lNKyxcM9cUm7vllPWjc00BAAAAAAAAAAAAAABLSNlidjdNWT9trik2N61s8ZG5pgAAAAAAAAAAAAAAgCWkbDG7aYWFW801xeZuM2X9/fMMAQAAAAAAAAAAAAAAy0jZYnbvn7J+63mGOIn7JunBda1fX7WYOAAAAAAAAAAAAAAAsDyULWb3jqyVF4YqyX3mH+VEVXWbJHceXY7dfsd80wAAAAAAAAAAAAAAwPJRtpjdlWPXowkS96yqs+YdZoIHb3LvdXNLAQAAAAAAAAAAAAAAS0rZYnavGpwPJ0eckuThc84yyTdtcu8Vc0sBAAAAAAAAAAAAAABLStlidi/Z5N4T55Zigqq6Z9YKH521IkgPbn84ySsXEAsAAAAAAAAAAAAAAJaKssWMuvuNSd41uszxUkMl+cyqesCisiX5sQlro3zP7+4jc84DAAAAAAAAAAAAAABLR9lie/44ayWGcZXkN6vqzDnnSVV9R5IvzPHix7hnzTcRAAAAAAAAAAAAAAAsJ2WL7Xl6kmPr58PpFkly5yS/Ms8wVfXAJE8ZZMjY+buTPHuemQAAAAAAAAAAAAAAYFkpW2xDd781yZ9n4wSJUeGiknxNVf1WVZ2y11mq6kuTPD/JqYMc45me2t3Hxj8LAAAAAAAAAAAAAACcSNli+/5HkiPr56MpEsPCxWOTPK+qPnkvHl5Vp1fV/0ryR0nOGjx3mCdJ3pXkF/YiAwAAAAAAAAAAAAAAHETKFtvU3Zcn+ZlsnCSRbCxcPCzJm6vqJ6vq1rvx3Ko6taq+PckVSX44az/DnvTW9fXv6e4jE+4DAAAAAAAAAAAAAAATKFvszI8leVmOFxtGhtenJ/mBJO+uqudX1TdV1b2r6oytPqSqbltVX11Vz0zyH0memuSCsecMp1qM1n+zu/98W98MAAAAAAAAAAAAAABWVHVPGorAVlXVhUlekbXyQ7Jx0sWkIsTw3vVJzsvGgsTo9S+S3H79OGf4yAl7TVp7ZZKHdPcNs30jOHiOXnulX3QAAAAAAPvYmRdetugIAAAAAABswdEjV9fJ33UwmGyxQ9397iQPzdrEiWR6CWJUpBgdh7JWtBi+b/j6RUk+Lcm5Y58b7ZXBWsbW3prk8xUtAAAAAAAAAAAAAABgdsoWu6C735rkQVkrOZysEDF+TDPtM+N7Zmz91Uke3N3v2+bXAQAAAAAAAAAAAACAlaZssUu6+21J7p/kz3NiUSLZOJ1iK6NTtvLZ8ff8cZKHdPd7t/UlAAAAAAAAAAAAAAAAZYvd1N0f7O4vTfI1Sa7J5OkUIycrXWxWzBgvWVyX5Ou7+yu6+0Pb/wYAAAAAAAAAAAAAAICyxR7o7mckuSTJjyT5z2wsTfSEY9PtJhyj/T6S5OeSfEp3//aufgkAAAAAAAAAAAAAAFhRyhZ7pLs/3N0/meQTk3x1kucnOZLJEysmFSrGp1cMj1cn+d4kt+vu7+vu9+/19wEAAAAAAAAAAAAAgFVxeNEBDrruPpLkmUmeWVVnJPmsJPdPcq8kd0lyuySnb7LFtUnenuQ1SV6e5IXdfdWehgYAAAAAAAAAAAAAgBWmbDFH3f3RJM9dPz6uqm6R5JwkZyQ5NcnHknwkyX+ufwYAAAAAAAAAAAAAAJgTZYt9oLuvS3LdonMAAAAAAAAAAAAAAADJoUUHAAAAAAAAAAAAAAAA2E+ULQAAAAAAAAAAAAAAAAaULQAAAAAAAAAAAAAAAAaULQAAAAAAAAAAAAAAAAaULQAAAAAAAAAAAAAAAAaULQAAAAAAAAAAAAAAAAaULQAAAAAAAAAAAAAAAAYOLzrAqqmqQ0numOTSJJck+YQk5yc5K8npSU5JcsP68b4k70lyVZI3JXljd39wAbEBAAAAAAAAAAAAAGBlKFvMQVVdkuS/JnlokgcmOXMHe70xyd8n+cskL+zuY7uREQAAAAAAAAAAAAAAWKNssUeq6tQkj0nybUnuM7y1w60vTXL39X2vrarfTfIL3X3VDvcFAAAAAAAAAAAAAACSHFp0gIOoqp6Q5O1JfjNrRYsaHL3DI4O9bpPkiUmuqKqnV9X5c/h6AAAAAAAAAAAAAABwoClb7KKqultVvSrJ05JcmMkFi2Rj+WLWY7x8UVmbUPL1Sf61qv7bXn9PAAAAAAAAAAAAAAA4yJQtdklVfWOSVya5V04sRYwXJnb0qLF9hs84O8kvV9VzquqcHT4HAAAAAAAAAAAAAABWkrLFLqiqJyf51SRn5HjRIplcrhifTDHrseHRmVy6+PwkL6mq2+3KFwQAAAAAAAAAAAAAgBWibLFDVfWzSb4vG6dZjJcsxssS45MutnpM2isT7leSeyT5m6q67e58UwAAAAAAAAAAAAAAWA2HFx1gmVXVdyf5npxYfBiZtH5dklckeV2S1yZ5a5IPJPng+nE0ybmD44Ikn7Z+3CvJXQZ799je44WLOyb5q6r6jO4+ss2vCQAAAAAAAAAAAAAAK6W6++Tv4gRV9elJXprjhZUae8uwCPGRJM9J8owkz+vuG3fw3Dsn+ZokX5nkkpxYuBg+fzRt45e6+zu3+0xYdkevvdIvOgAAAACAfezMCy9bdAQAAAAAALbg6JGrx/9u/cBSttiGqqokr0xy7xwvNYwMyw+d5HeT/FB3v2cPMnxDkv+V5LYnyXEsyf27+593MwMsC2ULAAAAAID9TdkCAAAAAGA5rFLZ4tCiAyypr8rJixZvy1rB4fG7XbRIkl7z9CR3TPLrOV7uGBnmOpTkybudAQAAAAAAAAAAAAAADiJli+35zglrw+LFPya5b3e/aq+DdPf13f1NSb47kwsXo+uHVNU99joPAAAAAAAAAAAAAAAsO2WLGVXVnZPcLxvLFaPzTvKmJJ/T3R+YZ67u/oUk/zMnFi6Gvm5ugQAAAAAAAAAAAAAAYEkpW8zukWPXw2LD0SSP7u4PzzHP8SDd/zvJ3+XEwsWoDPJ5i8gFAAAAAAAAAAAAAADLRNlidg+YsDYqN/xKd79lznnGfffYdQ3O71xV584xCwAAAAAAAAAAAAAALB1li9nddZN7T51biim6+/VJ/iYnTrcY2Sw/AAAAAAAAAAAAAACsPGWL2V2Q4yWGzvHJEa/t7ncsJNGJnr3JvQvmlgIAAAAAAAAAAAAAAJaQssXszpmw1klePu8gm3jZJvfOnlsKAAAAAAAAAAAAAABYQsoWszs2Zf1tc02xuSs2uTctPwAAAAAAAAAAAAAAEGWL7bh+xvVF2CzLfsoJAAAAAAAAAAAAAAD7jrLF7K6dsn6zuabY3GZZpuUHAAAAAAAAAAAAAACibLEdb05SE9ZvO+8gm7jNJvfePLcUAAAAAAAAAAAAAACwhJQtZve6KeuXzjXF5oZZenD+7u5+/5yzAAAAAAAAAAAAAADAUlG2mN3zxq47a5MuHlJV++Xf8xFj15W1nM9fQBYAAAAAAAAAAAAAAFgq+6UcsDS6+xVJ3j3h1nlJvmTOcU5QVTdL8phsnGgx8uw5xwEAAAAAAAAAAAAAgKWjbLE9v5i1aRFDleRHqmp8fd6+JcltJ6xfkeS5c84CAAAAAAAAAAAAAABLR9lie345yfsG16MpEp+a5AfnH2dNVd0hyY9l41SLWr/+qe6eNO0CAAAAAAAAAAAAAAAYULbYhu7+YJLvysbpFr1+/eNV9SXzzlRVt0rynCTnDpfXc724u39z3pkAAAAAAAAAAAAAAGAZKVtsU3c/I8nv5cTCxSlJnlVVT5hXlvWJFn+f5G7ZONUiSd6b5HHzygIAAAAAAAAAAAAAAMtO2WJnnpDkBTmxcHFqkqdV1Z9U1e336uFVdbiqvifJP+fEokUluT7JI7v7nXuVAQAAAAAAAAAAAAAADhplix3o7huTfGGS38+JhYtK8iVJ3lRVP70+fWJXVNXpVfVVSd6Y5ClJzht7fiV5V5LP7O7X7NZzAQAAAAAAAAAAAABgFVR3n/xdnFRVfVuS/zfJWcPl9dfRP/IrkzwryUuSvKG7b5hh/zskuVfWChyPWn/O+P6j62cn+ebuvmbGrwEH0tFrr/SLDgAAAABgHzvzwssWHQEAAAAAgC04euTqOvm7DoaVKltU1Y/u8SMuTvK4SY8enI/+wY8leVuStyb5QJLrk3wwydEk5yQ5d/24MMmlSc6esF+PrXWSf0nyF5PCdfePb/WLwEGibAEAAAAAsL8pWwAAAAAALAdliwOqqo5lY0Fhzx41OB8vRIzbSp7xz22259T9uvuULTwLDhxlCwAAAACA/U3ZAgAAAABgOaxS2eLwogMsyDx/wNOKF5PuTzPL56at+2NzAAAAAAAAAAAAAADYglUtW+x18WArRYgee93unuO2W+gAAAAAAAAAAAAAAACyumWL/VA+2KsM4/uaaAEAAAAAAAAAAAAAADM4tOgAAAAAAAAAAAAAAAAA+8mqTrYw7QEAAAAAAAAAAAAAAJhoFcsWtegAAAAAAAAAAAAAAADA/rVSZYvuPrToDAAAAAAAAAAAAAAAwP6mfAAAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADCgbAEAAAAAAAAAAAAAADBweNEBVkFVnZ7k7kkuTXJxkguS3CrJGUlOS3LKHGJc3t3fMofnAAAAAAAAAAAAAADAUlO22CNVdVGSr07yuUkekOTUxSbK2Qt+PgAAAAAAAAAAAAAALAVli11WVfdL8qNZK1nUaHlxiQAAAAAAAAAAAAAAgFkoW+ySqrplkp/P2jSLZGPBouce6DhFDwAAAAAAAAAAAAAAmIGyxS6oqnsmeU6ST8zxcsN4wWIRpYdFljwAAAAAAAAAAAAAAGApKVvsUFU9MMnzk5y1vjQsOGw23WJa+WJSQWKrRY1Jn1W4AAAAAAAAAAAAAACAGShb7EBVfUqSv8xa0WJUatisYDF+f+K2Ez6/1aLGIqZnAAAAAAAAAAAAAADAgaJssU1VdTjJM5PcPJsXLUZrR5K8Z/24eZI7rb+nxl5ftH5+XpJbJLllkrPH9h3uPTy/Mck/JTk2IfLlM31BAAAAAAAAAAAAAABYUcoW2/etSe6TE0sVw+ubkvxukj9O8rfdfSRJquoJSZ4+adPu/qzxtaq6TZIHJvmMJJ+f5K5jzxo5JWtFi8d197tm/0oAAAAAAAAAAAAAAMChRQdYRlV1ZpL/mc2LFv+c5D7d/YTuft6oaLEd3X1Nd/9Fd//37r57ks9N8txMnqTxoCSvqaoHb/d5AAAAAAAAAAAAAACwypQttucxSW61fj4sWozOX5Tks7r7dXvx8O5+QXd/QZJHJXlvNpYuKsktkjyvqr54L54PAAAAAAAAAAAAAAAHmbLF9jx+7LoH5+9I8sju/tBeh+juv0xyjyR/m42lj05yWpJnVNX99zoHAAAAAAAAAAAAAAAcJMoWM6qqWyW5fzYWLJK1skMn+ebu/si88nT3tUkemeT52TjhopOcnuTZVXWLeeUBAAAAAAAAAAAAAIBlp2wxu4fk+L/bqGAxev2H7v6/8w7U3UeSPDrJ5RNun5/kyfNNBAAAAAAAAAAAAAAAy0vZYnb33uTeb8wtxZju/lCSx48vZ60I8viqutv8UwEAAAAAAAAAAAAAwPJRtpjdpYPzHpzflOTP5pxlg+5+WZJnZ61gMVRJvmP+iQAAAAAAAAAAAAAAYPkoW8zudmPXo2LDm7v7ozvdvKoO73CLXxi7Hk23eMwu7A0AAAAAAAAAAAAAAAeessXsLszGiRZZv37VLu1/6k4+3N0vSvLeCbfOTvKZO9kbAAAAAAAAAAAAAABWgbLF7M6asv7uGfY4tsm9s2fYZ5qX5PjEjaGH7MLeAAAAAAAAAAAAAABwoClbzO6MKesfmGGPj21y75wZ9pnmLVPW77YLewMAAAAAAAAAAAAAwIGmbDG7o1PWPzjDHpuVLc6fYZ9p/mPCWiW58y7sDQAAAAAAAAAAAAAAB5qyxeymlSpOn2GP6za5d8EM+0wzXubo9dfb7sLeAAAAAAAAAAAAAABwoClbzO76KevnzbDHNZvcu2SGfaY5Z8r62buwNwAAAAAAAAAAAAAAHGjKFrN7f5KasD5L2eLfNrl36UxpJrv1lPXTdmFvAAAAAAAAAAAAAAA40JQtZnfFlPVbbnWD7r4uyXWjy8FrJbnf9qN93L2nrH9gF/YGAAAAAAAAAAAAAIADTdlidm+ZsFZJ7j7jPm/M8QkZw0kZd6qq220nWJJU1WlJHpDjJY6h9213XwAAAAAAAAAAAAAAWBXKFrMbL1uMSg2zli1evsm9r5xxr6GvSnLO+vmwzNFJrt3BvgAAAAAAAAAAAAAAsBKULWb3hsH5cCLFGVV1xxn2+YcJa72+53euT6iYSVWdkeSHM3mqRZK8ZtY9AQAAAAAAAAAAAABg1ShbzKi735DkfaPLsdv3n2GrFyb56GCfYXHjE5P82jbi/VKST1k/rwn3/24bewIAAAAAAAAAAAAAwEpRttief8jkMsMXbXWD7r4hyV+M7VM5Xrz42qr62ao69WR7VdXNquq3knxdNhY3hmWQY0n+fqv5AAAAAAAAAAAAAABgVSlbbM/4hIhRweFzqupmM+zzyxPWhoWL707yqqp6fFXd/IQ3Vt2mqr4hyeVJHjvlGaP9/qS7/3OGbAAAAAAAAAAAAAAAsJKqu0/+LjaoqjsleUuOlyKGr4/q7r+aYa+/TfKQbJxIkZw4oeKmJP+R5D3r57dN8knr76kJnxled5J7dvfrt5oLDpKj117pFx0AAAAAwD525oWXLToCAAAAAABbcPTI1XXydx0MJltsQ3e/Nclrc7zIMPSNM273vUmOjrYerA/3riSHk3xikvskuV+Si7P28xu9b7OixTMULQAAAAAAAAAAAAAAYGuULbbvD8euR+WGL6iqO291k+5+TZIfz8aixMiwSDE6hs8bro0XLUavb0ny37aaBwAAAAAAAAAAAAAAVp2yxfY9c/21Bkey9m/6/bNs1N3/O8nvZ2O5YmS4d7Z4f7R+XZIv7e6PzJIHAAAAAAAAAAAAAABW2eFFB1hW3f3OqvqaJOdMuP2xbWz5uCTXJ/mWTJ5YMWnyxYZIg/NK8q4kX9Ddl28jCwAAAAAAAAAAAAAArCxlix3o7mfs4l6d5Nuq6u+T/EKS83PiFIuTGRUyfj/Jd3X3dbuVDwAAAAAAAAAAAAAAVsWhRQdgo+7+4yR3SPJ9Sd6UtQLFVo6PJnlGkvt292MVLQAAAAAAAAAAAAAAYHtqbaAC+1VVXZLksiR3TfLJSc5NcrOslSuuSXJFklcmeXF3f3RROWE/O3rtlX7RAQAAAADsY2deeNmiIwAAAAAAsAVHj1xdi84wL4cXHYDNdfcVWStUAAAAAAAAAAAAAAAAc3Bo0QEAAAAAAAAAAAAAAAD2E2ULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAgcOLDjBPVfXYRWdYpO7+3UVnAAAAAAAAAAAAAACA/W6lyhZJfjtJLzrEAilbAAAAAAAAAAAAAADASaxa2WKkFh1gAVa5ZAIAAAAAAAAAAAAAAFu2qmWLVSserGK5BAAAAAAAAAAAAAAAtmVVyxarVD5YtWIJAAAAAAAAAAAAAADsyKFFBwAAAAAAAAAAAAAAANhPVnWyhWkPAAAAAAAAAAAAAADARKtYtqhFBwAAAAAAAAAAAAAAAPavVStb3H7RAQAAAAAAAAAAAAAAgP1tpcoW3f3ORWcAAAAAAAAAAAAAAAD2t0OLDgAAAAAAAAAAAAAAALCfKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMKFsAAAAAAAAAAAAAAAAMHF50gHmqqisXnWGBursvWXQIAAAAAAAAAAAAAADY71aqbJHk4iSdpBacYxF60QEAAAAAAAAAAAAAAGAZrFrZYmTVigerWC4BAAAAAAAAAAAAAIBtObToAAAAAAAAAAAAAAAAAPvJqk62WKVJD6s2xQMAAAAAAAAAAAAAAHZkVcsWe11A2GqZY6s5ZimHKFcAAAAAAAAAAAAAAMAOrGLZYpFTLaYVITbL1BM+t9n7V2lqBwAAAAAAAAAAAAAA7LpVK1v8zh7vf26SL8rxwsPodViWGJYhbkhyZZIPJPng+nF0fZ/RcUGS88eeM75fD15fn+Q1O/weAAAAAAAAAAAAAACwslaqbNHdj9+rvavqkUl+NRuLFj04T5L3JvnjJC9N8tokl3f3sS3sfdskn5bk3km+NMl9128Np16MChd3T/LCJD/S3Tfs4CsBAAAAAAAAAAAAAMBKqu4++bvYVFU9KcmPji7XX3tw/pwkv5Lkhd190y4875Ikj03yXVmbfjFp0sU/J3lUd79np8+DZXf02iv9ogMAAAAA2MfOvPCyRUcAAAAAAGALjh65uk7+roPh0KIDLLuqenrWihaV40WHUdHiDUke3t1f3N3P342iRZJ09xXd/aQkd0zyG9lYthg9+9OTvLyqPmk3ngkAAAAAAAAAAAAAAKtC2WIHqurHkzwhJ5YsKsnPJ7lXd//tXj2/u6/p7m9M8ogk7xveWn+9KMnzquoWe5UBAAAAAAAAAAAAAAAOGmWLbaqqhyb5kRwvWSTHSxc/3N1P3K1JFifT3X+X5MFJ3jtcXn+9c5JfnUcOAAAAAAAAAAAAAAA4CJQttqGqTkvy6+PLWSs4/Hp3/9S8M3X3m5N8aZIbh8vruf5rVX3hvDMBAAAAAAAAAAAAAMAyUrbYnscluTjHp0eMXJXku+aeZl13/1OSn8tawWKokjxp/okAAAAAAAAAAAAAAGD5KFtszxOzsWgxmmrxE919w2IifdxPJrl+cD3Kea+qetAC8gAAAAAAAAAAAAAAwFJRtphRVd09yZ0m3Lohye/POc4Juvv6JH+UE6dbJMmXzTkOAAAAAAAAAAAAAAAsHWWL2X3+2PVoqsWL9sFUi5G/HrvurOX8vAVkAQAAAAAAAAAAAACApaJsMbt7Tln/lzlmOJnXDs6HEy4uqaqz5h0GAAAAAAAAAAAAAACWibLF7O42Zf3f55pic+/e5N5d55YCAAAAAAAAAAAAAACWkLLF7M5P0hPWPzrvIJu4YZN7588tBQAAAAAAAAAAAAAALCFli9mdM2X97Lmm2NxZm9yblh8AAAAAAAAAAAAAAIiyxXbcbMr6xfMMcRIXb3JvWn4AAAAAAAAAAAAAACDKFttx/ZT1+8w1xeY+fZN7H5pbCgAAAAAAAAAAAAAAWELKFrO7Zuy6k1SS+1XVbRaQZ5JHbXLvvXNLAQAAAAAAAAAAAAAAS0jZYnZvzlq5IoPXJDklyTfPP85GVfVJSb4gayWQSd4yxzgAAAAAAAAAAAAAALB0lC1m96oJa6PpFt9XVZ8w5zzjfjrJ4fXzysbSxVXdfe38IwEAAAAAAAAAAAAAwPJQtpjdX49dD6dbnJPkGVW1kH/Xqnpskq/I8fLHx2+trz1vEbkAAAAAAAAAAAAAAGCZKFvMqLtfneTy0eX663CCxIOzVrg4ZZ65quoLkzwtGydZjPuDOcUBAAAAAAAAAAAAAIClpWyxPb+UjZMjkuOFi0ry6CQvrKpPmEeYqvreJH+a5GaDLBnk6SSv7e4XzyMPAAAAAAAAAAAAAAAsM2WL7fm1JFesnw8nSQwLFw9O8taq+qGqOm0vQlTVg6vq1UmenOTw4NmT/NBeZAAAAAAAAAAAAAAAgING2WIbuvtoku8cLg3Oa3B9dpKfyFrp4v+pqrvs9NlVdXZVfW1VPT/J3yb5tLFnDjON1v+su5+/02cDAAAAAAAAAAAAAMAqqO7xv9Fnq6rqZ5J8byZPlBj9w9bY9euTvCTJ65K8Nslbk3ygu49N2P+sJBcm+dSslSruleShSU6fsvek67cnuXd3f2DGrwcHxtFrr/SLDgAAAABgHzvzwssWHQEAAAAAgC04euTq8b+bP7AOLzrAkvuBJBcn+bKcWHgYTZUYX//UJPcY36iqbkjywSRHk5yTtakYkyaPDP/jHN97fO2aJI9UtAAAAAAAAAAAAAAAgK2b9Mf8bNH6NIrHJPnTnDhVIutrw/UeWx8eZyQ5P8ntkpyX5JQp7+vBMdx/+OxK8u9JPru7L9/5NwUAAAAAAAAAAAAAgNWhbLFD3X20ux+d5MnD5UwuXYyXJbZzDPfbEGVw71+SPKC7X7vjLwgAAAAAAAAAAAAAACtG2WKXdPcPJvm8rE2UmDTNYmTStIrx4sRW3zs+5aKT/GyS+3f3VTv7RgAAAAAAAAAAAAAAsJqULXZRd78gyV2S/FSSj+XE0sV48WJoqwWMaVMu/i7Jp3f393f30R18DQAAAAAAAAAAAAAAWGnKFrusuz/U3T+c5OIkP5HkmmwsT/QOjwz2O5bkz5M8qLsf1t3/sqdfDgAAAAAAAAAAAAAAVkB1bzZsgZ2qqlOSPDTJlyV5WJJLNnn78IcxbbLF9UlenOT/JPnT7r52N3LCQXb02iv9ogMAAAAA2MfOvPCyRUcAAAAAAGALjh65etrfuR84hxcd4KDr7puS/N/1I1V1YZJ7Jrl71ooXFyQ5P8mZSU7P2rSRjyW5Icl/Jvn3JFcleVOSNyR5fXcfm+uXAAAAAAAAAAAAAACAFaJsMWfd/e4k707y3EVnAQAAAAAAAAAAAAAATnRo0QEAAAAAAAAAAAAAAAD2E2ULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAgcOLDrDKqupQkrOSnJHktCQ1utfd71pULgAAAAAAAAAAAAAAWGXKFnNSVZcmeXCSeyW5NMntkpyfydNFOn42AAAAAAAAAAAAAACwEP6gfw9V1T2SPCHJo5N8wvDWHjznvlNuv767X7mbzwMAAAAAAAAAAAAAgINM2WIPVNUDkvxYkoePlia8rad9fBuP/HCSX8vkKRmvTXLvbewJAAAAAAAAAAAAAAAradIf57NNVXXzqvrNJC/NWtGi1o+ecHz8Y4NjW7r7yiTPHNtrdHxaVX3qdvcGAAAAAAAAAAAAAIBVo2yxS6rqPlmbIvG4nFiySCYXIbZdsJjg59dfJxU6HruLzwEAAAAAAAAAAAAAgANN2WIXVNUXJnlxkouysWQxXqrYbMLFjnT3q5O8LBsLHKMMX11Vu1nsAAAAAAAAAAAAAACAA0vZYoeq6pFJ/iTJ6TmxZJGcWKzYq8kWSfJ7w2iD89sm+fRdfhYAAAAAAAAAAAAAABxIhxcdYJlV1V2TPDPJqdlYphgZX7s6axMw3pnkfUnukeRrc7ygsVPPSvLUJKfkxKkZD0/yyl14BgAAAAAAAAAAAAAAHGgmW2xTVR1O8kdJzsmJpYrhhIvrkzw5yR27+6Lufkx3/1B3PyXJi3YzU3dfl+SlmVzcePhuPgsAAAAAAAAAAAAAAA4qZYvt++9J7p7JRYvR9a8l+aTu/sHuvmJOuf567HpU+vgvVXXanDIAAAAAAAAAAAAAAMDSUrbYhqo6L2tli2GxYjjN4mNJvqq7v6W7PzjneC8enA8nXJyW5B5zzgIAAAAAAAAAAAAAAEtH2WJ7vjnJuevno6LF6PxYkq/p7j9cRLAkr05ydP28x+7dZc5ZAAAAAAAAAAAAAABg6ShbbM9jc2KRYVS6eFJ3/9n8I63p7o8luWLKbWULAAAAAAAAAAAAAAA4CWWLGVXV3ZLcdXSZjaWLdyZ58txDnejyrGUbp2wBAAAAAAAAAAAAAAAnoWwxuwdNWBuVLn6su4/OOc8k/zZhrZJcNO8gAAAAAAAAAAAAAACwbJQtZveAwflwqsWNSf5szlmmec/Y9SjnufMOAgAAAAAAAAAAAAAAy0bZYnafMnY9mmrxsu6+fgF5JvnglPVz5poCAAAAAAAAAAAAAACWkLLF7D45GydajPzLnHNs5mNT1k22AAAAAAAAAAAAAACAk1C2mN20wsI1c02xuVOnrJ8x1xQAAAAAAAAAAAAAALCElC1md+aU9WvnmmJzt5yyPm3iBQAAAAAAAAAAAAAAsE7ZYnZHp6xPK2EswrSyxUfnmgIAAAAAAAAAAAAAAJaQssXsPjxl/VZzTbG586esv2+uKQAAAAAAAAAAAAAAYAkpW8zuuinrt51ris09IEkPrmv9+qrFxAEAAAAAAAAAAAAAgOWhbDG7t2etvDBUSe63gCwnqKrbJbl4dDl2+x1zDQMAAAAAAAAAAAAAAEvo8KIDLKErkjxicN1ZKzXcvapu0d3TJl/My0M3ufeauaWAfeTs2z140REAAAAAANhEn/wtAAAAAAAwVyZbzO7lg/MaO3/knLNM8q2b3HvF3FIAAAAAAAAAAAAAAMCSUraY3UunrFeS75tnkBMCVD04yf1yfNrG8P8I6rokr15ELgAAAAAAAAAAAAAAWCbKFjPq7rclecvoMhtLDZ9aVQuZblFVh5L8xKRbWcv3l919bL6pAAAAAAAAAAAAAABg+ShbbM8fZa3EMDQqXjytqm49/0j5iSSfMcgx7hnzjQMAAAAAAAAAAAAAAMtJ2WJ7np7kyPr5eLnhgiS/V1WH5xWmqr4syQ/k+ISNjJ3/a3e/YF55AAAAAAAAAAAAAABgmSlbbEN3X53k97OxZFE5Xrz47CTPrapz9jpLVT0xybPGcoxnespe5wAAAAAAAAAAAAAAgINC2WL7fjTJ9evnoykSw8LFw5K8pKoesBcPr6oLqur3kvxMklMGzx/lGWV5Y5Lf2IsMAAAAAAAAAAAAAABwEClbbFN3vztrhYsauzUsXNwjyUur6hlVdc/deG5VfUJVPSXJ25I8ZvC8j0cbnB9L8q3dPVwDAAAAAAAAAAAAAAA2cXjRAZZZdz+1qh6a5AtzvGCRbCxcVJKvSPIVVfW2JH+a5FVJ3pTk1JM9o6pukeSOSR6R5AuS3Hewb7JxqkYG553kp7v7Jdv9fgAAAAAAAAAAAAAAsIrK0IOdqapzk7w0yd2zsXCRTC5CjP+Dj98bFSVem+T2Sc6d8v6T7f1XSb64u4+d/FvAwXba6Rf5RQcAAAAAsI/ddMz/nAEAAAAAsAxuPHJ1nfxdB8OhRQdYdt39wSQPS/KWHC9KDIsQw7XhtIvhdIoM3j96vWeS8ya8f3z/j0cZrP1jki9XtAAAAAAAAAAAAAAAgNkpW+yC7n5vksuSvDgnTp5INhYreuzYdOsp7x8vagzX/zrJ53T3DTN/EQAAAAAAAAAAAAAAQNlit3T3+7I24eKpOV5+GC9UjE+p2GyEynixYtJnhtMykuRnkjyquz+y7S8CAAAAAAAAAAAAAAArTtliF3X3jd39PUkenORNmTzNYqumlTEmTbl4a5KHd/cPdPdN28kOAAAAAAAAAAAAAACsUbbYA9390iSfmuRrkrwxG4sTvcmxYZtN3jPa76ok35nk0u7+uz36OgAAAAAAAAAAAAAAsFKqe5ZhC2xHVT0gyeOSPDLJRWO3T/YDGJ9u8eEkz0/yB0n+oruP7UpIOMBOO/0iv+gAAAAAAPaxm475nzsAAAAAAJbBjUeuHv/79gNL2WLOqupuSe6f5F5J7pK18sWFSc7OxmLFTUnel+RdSd6e5DVJXp7kH7v7Y/PMDMtO2QIAAAAAYH9TtgAAAAAAWA7KFixEVZ2W5NQkH+vuo4vOAweFsgUAAAAAwP6mbAEAAAAAsBxWqWxxeNEBOG59YoWpFQAAAAAAAAAAAAAAsECHFh0AAAAAAAAAAAAAAABgP1G2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGFC2AAAAAAAAAAAAAAAAGDi86ADLqKoeO+XWS7v7irmGGVNVlyT5jEn3uvt35xwHAAAAAAAAAAAAAACWjrLF9vx2kp6w/o1JFlq2SPKQJE+bck/ZAgAAAAAAAAAAAAAATkLZYmdqcD6pfLEoNWFtP+UDAAAAAAAAAAAAAIB9S9liZ0YFhknlhkUbliv2Yz4AAAAAAAAAAAAAANiXDi06wJLb7yWG/Z4PAAAAAAAAAAAAAAD2HWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWULAAAAAAAAAAAAAACAAWWLg2f4M+0p5wAAAAAAAAAAAAAAwBTKFgfPWVPWb5xrCgAAAAAAAAAAAAAAWFLKFgfPeVPWPzrXFAAAAAAAAAAAAAAAsKSULQ6eO01Zf99cUwAAAAAAAAAAAAAAwJJStjh4HpCkB9e1fv2excQBAAAAAAAAAAAAAIDlomxxgFTVZUluP7ocu33FnOMAAAAAAAAAAAAAAMBSUrY4IKrqgiRP2+Qtr5tXFgAAAAAAAAAAAAAAWGaHFx2A7auq2yW5W5LPT/LYJOcm6Zw41SJJXjrHaAAAAAAAAAAAAAAAsLSquxedYaGq6sptfOziHC81DF/fl+RDuxZuslOTnJHk7PXzkfGCRQ/Wr0nyCb3qP2xW1mmnX+S/fQAAAACAfeymY8cWHQEAAAAAgC248cjVkwYDHEgmW2wsTsyiJrzeev1YhGG5IoPzTvIHihYAAAAAAAAAAAAAALA1yhbHzVJGmFbMWHShYZRrmOOjSX5uAVkAAAAAAAAAAAAAAGApKVsst5NN4+gk/7O7/20eYQAAAAAAAAAAAAAA4CBQtjjuZMWFee2xXcNpFqMcP9/d/98iwgAAAAAAAAAAAAAAwLJStjiuT/6Wj5tWqphlj70wyvXvSb63u5+1yDAAAAAAAAAAAAAAALCMlC3W7NZEikVOtvhwkn9I8odJ/rC7jywwCwAAAAAAAAAAAAAALC1li+R3tvGZx2VtikWNvf5Tkn/dvWgT3ZTkSJLrk1yT5Kokb0nyhu6+aY+fDQAAAAAAAAAAAAAAB15196IzLJ2qOpbJZYtv7O7fXGQ24ESnnX6RX3QAAAAAAPvYTceOLToCAAAAAABbcOORq2vRGebl0KIDAAAAAAAAAAAAAAAA7CfKFgAAAAAAAAAAAAAAAAPKFjvTY68AAAAAAAAAAAAAAMCSO7zoAEusFh0AAAAAAAAAAAAAAADYfcoW23P7KevXzjUFAAAAAAAAAAAAAACw65QttqG737noDAAAAAAAAAAAAAAAwN44tOgAAAAAAAAAAAAAAAAA+4myBQAAAAAAAAAAAAAAwICyBQAAAAAAAAAAAAAAwICyBQAAAAAAAAAAAAAAwICyBQAAAAAAAAAAAAAAwMDhRQdYVlV12yRfOeX2e7r7j+ac59FJLphy+7e6+/p55gEAAAAAAAAAAAAAgGWlbLF935zkSVPuPXGeQdZdmOTnpty7KckvzTELAAAAAAAAAAAAAAAsreruRWdYOlV1KMnbk1w04fZ7k1zc3TfMOdMZSd6R5DYTbr+xu+8xzzywn5x2+kV+0QEAAAAA7GM3HTu26AgAAAAAAGzBjUeurkVnmJdDiw6wpD43a0WLHhxZf33qvIsWSdLdH03y1EGOYa67VdUD550JAAAAAAAAAAAAAACWkbLF9jxqynon+e055hj32zlesBj3JXPMAQAAAAAAAAAAAAAAS0vZYns+JxtLDbV+/ZLu/vfFREq6+91JXryeZ6iyNo0DAAAAAAAAAAAAAAA4CWWLGVXVnZJ88uhy7Paz5hxnkj8cnI9KIElyt6q6cAF5AAAAAAAAAAAAAABgqShbzO7+m9z7m7mlmO6Fm9x74NxSAAAAAAAAAAAAAADAklK2mN1dB+c9OP9Ad//rvMOMW8/wgdHl2O27BgAAAAAAAAAAAAAA2JSyxezuNnZdWSs1vHoBWab556zlGqdsAQAAAAAAAAAAAAAAJ6FsMbs7TFl/y1xTbO7yCWuV5I7zDgIAAAAAAAAAAAAAAMtG2WJ252VtksW46+YdZBP/OXY9ynuLeQcBAAAAAAAAAAAAAIBlo2wxu3OmrI8XHBZpWpaz55oCofZE6QAAmUdJREFUAAAAAAAAAAAAAACWkLLF7KaVLT401xSb+/CU9WnZAQAAAAAAAAAAAACAdcoWs7txyvoZc02xudOnrN9srikAAAAAAAAAAAAAAGAJKVvM7iNT1m811xSbu+WU9Y/ONQUAAAAAAAAAAAAAACwhZYvZvX/K+u3nGeIk7jBl/UNzTQEAAAAAAAAAAAAAAEtI2WJ270pSY2uV5D4LyDLNvZP04HqU9+oFZAEAAAAAAAAAAAAAgKWibDG7d4xdj0oNd66q2845ywmq6jZJ7jrhVid5+5zjAAAAAAAAAAAAAADA0lG2mN1rBuc1dv7oOWeZ5MtzPNf4BI43zjkLAAAAAAAAAAAAAAAsHWWL2b1swlpnrdjw7XPOskFVVZJvy/FpG+P+aY5xAAAAAAAAAAAAAABgKSlbzO7VSd6/fj4qWYzcqaoeN/dEx31tkrusn1c2li6OJPnHuScCAAAAAAAAAAAAAIAlo2wxo+6+MclzsrFkkRwvXjylqi6Yd671Zz4lJ061GJUuXtDdH5p3LgAAAAAAAAAAAAAAWDbKFtvz+2PXw+LFrZI8u6rOnFeYqjojyZ8lufWEPCPjmQEAAAAAAAAAAAAAgAmULbahu1+Y5A2jy/XXGpzfN8lfV9XN9zpLVZ2X5LlJ7p/j0zWGuZLknUn+dK+zAAAAAAAAAAAAAADAQaBssX0/mRMnSIwKF5XksiRvqKpH7FWAqnpYktcnedC0t6zn+anuPrZXOQAAAAAAAAAAAAAA4CBRttim7n5Wkn8cXQ5uDSdcXJjkeVX1q1X1Kbv17Kq6Q1X9SpLnJ7ldNpY8MjjvJK9J8vTdejYAAAAAAAAAAAAAABx01d0nfxcTVdXdkrwiyRmjpcHtHqz1+vGCrBUf/r67r5vxWbdI8uAk35Tks9f3HRY7hkWL0fXHkjywu18zy7PgoDnt9Iv8ogMAAAAA2MduOmZANwAAAADAMrjxyNV18ncdDMoWO1RVj0/yG9k4WWJkWhEiSd6W5GVJLk/y/sHRSW6R5Obrx52SPCDJHYePnbL/aG1UwvjW7v61Gb8SHDjKFgAAAAAA+5uyBQAAAADAclilssXhRQdYdt39W1V1SZIfzonlh81KEXdM8ilbfMy0Esf4veH6UxQtAAAAAAAAAAAAAABgdsoWu6C7f6SqTk/yxKwVHsanXNRgPWPrW3rEhLXNpmj8bHf/wBb3BgAAAAAAAAAAAAAABg4tOsBB0d3fl+Q7k4zmXI+XK2rsGL7nZMe0zw/3qCQ3Jfm27v7+XfxqAAAAAAAAAAAAAACwUpQtdlF3/2KSy5K8LScWKsaNlydOdpzwuGwsYrw5yWd096/sxncBAAAAAAAAAAAAAIBVpWyxy7r7ZUk+LcmTklyf6VMsZt56wucryQeS/FCSe3f3K7efHAAAAAAAAAAAAAAASJLq3s7f/bMVVXXLJN+S5BuSfPLg1nb/0YcTLt6W5OlJntbdH9jmfrASTjv9Ir/oAAAAAAD2sZuOHVt0BAAAAAAAtuDGI1fXyd91MChbzEFVVZIHJvmCJI9Ico8kp864zZEkr0vygiR/uT5BA9gCZQsAAAAAgP1N2QIAAAAAYDkoW7CnqurUJJcmuSTJRUnOT3JmkjPW3/LRJB9O8h9JrkpyRZI3dvfR+aeF5adsAQAAAACwvylbAAAAAAAsB2ULgANE2QIAAAAAYH9TtgAAAAAAWA6rVLY4tOgAAAAAAAAAAAAAAAAA+4myBQAAAAAAAAAAAAAAwICyBQAAAAAAAAAAAAAAwICyBQAAAAAAAAAA/P/s3XeYXVWhPuBvJwFCKtJBQi/Su1IUBKSDKEWKIgre3wVs1yuiqFdFketV8IoIXEVEpINYUIqIqEhRQIpIb6H3lkASUmb//khGdiYzZ+bMnDlnyvs+zzxzzl5r1vp28JnH7Mw3CwAAACqULQAAAAAAAAAAAAAAACqULQAAAAAAAAAAAAAAACqULQAAAAAAAAAAAAAAACqULQAAAAAAAAAAAAAAACqULQAAAAAAAAAAAAAAACqULQAAAAAAAAAAAAAAACqULQAAAAAAAAAAAAAAACpGtTpAMxVF8XAPppVlWa7WgHUGmm7vCwAAAAAAAAAAAAAAGGZliyQrJymTFDXmlA1aZ6DpyX0BAAAAAAAAAAAAAMCwN9zKFu26Kh7UW54YLAWGwVQKAQAAAAAAAAAAAACAlhrR6gAAAAAAAAAAAAAAAAADyXA92aKzkx56c0rFYDgxYrCcvgEAAAAAAAAAAAAAAAPCcC1bNKqAoMgAAAAAAAAAAAAAAABDzHAsWzTqNIrBcKoFAAAAAAAAAAAAAABQp+FWtjhrgK0DAAAAAAAAAAAAAAAMMEVZlq3OANCvFhk9yTc6AAAAAIABbE5bW6sjAAAAAADQA7NnPlm0OkOzjGh1AAAAAAAAAAAAAAAAgIFE2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBC2QIAAAAAAAAAAAAAAKBiVKsDNFNRFA+3OkMLlWVZrtbqEAAAAAAAAAAAAAAAMNANq7JFkpWTlEmKFudohbLVAQAAAAAAAAAAAAAAYDAYbmWLdsOteDAcyyUAAAAAAAAAAAAAANArI1odAAAAAAAAAAAAAAAAYCAZridbDKeTHobbKR4AAAAAAAAAAAAAANAnw7Vs0d8FhJ6WOXqao55yiHIFAAAAAAAAAAAAAAD0wXAsW7TyVIuuihC1MpWdfF2t+cPp1A4AAAAAAAAAAAAAAGi44Va2OKuf15+QZK+8WXho/1wtS1TLEDOSPJzk1SRT5n3MmrdO+8dySZbpsE/H9crK5zuT3NbH+wAAAAAAAAAAAAAAgGFrWJUtyrL8aH+tXRTFbkn+L/MXLcrK6yR5LsnFSa5PckeS+8qybOvB2ksn2TDJJkn2TrL5vKHqqRfthYt1k1yd5MtlWc7owy0BAAAAAAAAAAAAAMCwVJRl2f0saiqK4qtJvtL+dt7nsvL60iSnJbm6LMs5DdhvtSQfTvLpzD39orOTLv6e5L1lWT7T1/1gsFtk9CTf6AAAAAAABrA5bd3+bioAAAAAAAaA2TOfLLqfNTQoW/RRURSnJzk085csMu/9P5P8R1mW1/TT3kslOT7JRyv7p/L68STvKsvysf7YHwYLZQsAAAAAgIFN2QIAAAAAYHAYTmWLEa0OMJgVRfH1JIflzdMk2k+zKJJ8L8nG/VW0SJKyLJ8vy/LfkuyY5MXq0LzPk5JcWRTFW/orAwAAAAAAAAAAAAAADDXKFr1UFMX2Sb6cN0sWyZuliy+WZfmfZVnOaUaWsiz/mGTbJM9VL8/7vFaS/2tGDgAAAAAAAAAAAAAAGAqULXqhKIpFkvy44+XMLTj8uCzLbzU7U1mW9yTZO8ns6uV5ufYtimLPZmcCAAAAAAAAAAAAAIDBSNmidw5JsnLePD2i3eNJPt30NPOUZXljku9mbsGiqkjy1eYnAgAAAAAAAAAAAACAwUfZonf+M/MXLdpPtTiuLMsZrYn0L8cnmVp5355z46IotmlBHgAAAAAAAAAAAAAAGFSULepUFMW6SdbsZGhGknOaHGcBZVlOTXJRFjzdIkn2aXIcAAAAAAAAAAAAAAAYdJQt6rd7h/ftp1pcOwBOtWh3RYf3Zebm3LUFWQAAAAAAAAAAAAAAYFBRtqjfRl1cv72JGbpzR+V19YSL1YqiGNvsMAAAAAAAAAAAAAAAMJgoW9RvnS6uP93UFLU9VWNs7aalAAAAAAAAAAAAAACAQUjZon7LJCk7uT692UFqmFFjbJmmpQAAAAAAAAAAAAAAgEFI2aJ+47u4Pq6pKWobW2Osq/wAAAAAAAAAAAAAAECULXpj4S6ur9zMEN1YucZYV/kBAAAAAAAAAAAAAIAoW/TG1C6ub9rUFLVtVmPstaalAAAAAAAAAAAAAACAQUjZon7Pd3hfJimSvL0oiqVakKcz760x9lzTUgAAAAAAAAAAAAAAwCCkbFG/ezK3XJHK5yQZmeTw5seZX1EUKybZI3NLIJ25t4lxAAAAAAAAAAAAAABg0FG2qN8tnVxrP93iqKIolm1yno7+J8moea+LzF+6eLwsyxeaHwkAAAAAAAAAAAAAAAYPZYv6XdHhffV0i/FJziuKoiV/rkVRfDjJ/nmz/PGvoXnXrmxFLgAAAAAAAAAAAAAAGEyULepUluWtSe5rfzvvc/UEiW0zt3Axspm5iqLYM8mPMv9JFh2d26Q4AAAAAAAAAAAAAAAwaClb9M4pmf/kiOTNwkWRZL8kVxdFsWwzwhRF8dkklyRZuJIllTxlkjvKsvxLM/IAAAAAAAAAAAAAAMBgpmzROz9M8tC819WTJKqFi22T3F8UxTFFUSzSHyGKoti2KIpbk3w7yajK3p05pj8yAAAAAAAAAAAAAADAUKNs0QtlWc5K8qnqpcrrovJ+XJLjMrd08bWiKN7W172LohhXFMXBRVH8Lsk1STbssGc1U/v1X5Rl+bu+7g0AAAAAAAAAAAAAAMNBUZYdf0afniqK4jtJPpvOT5Ro/4MtOry/M8l1Sf6R5I4k9yd5tSzLtk7WH5tk+SQbZG6pYuMk2ycZ3cXanb1/JMkmZVm+WuftwZCxyOhJvtEBAAAAAAxgc9oW+GcSAAAAAAAGoNkzn+z4c/NDlrJFHxRFMSLJhUn2yYKFh2TBEy86u95uRpIpSWYlGZ+5p2J0dvJIZ+t0de35JNuUZXlf13cBQ5+yBQAAAADAwKZsAQAAAAAwOAynskVnP8xPD807jeKgJJdkwVMlMu9a9XrZ4Xr1Y9EkyyRZIcnEJCO7mFdWPqrrV/cukjydZCdFCwAAAAAAAAAAAAAAqI+yRR+VZTmrLMv9kny7ejmdly46liV681Fdb74olbHbk2xRluUdfb5BAAAAAAAAAAAAAAAYZpQtGqQsyy8k2TVzT5To7DSLdp2dVtHVUSrdze14ykWZ5MQk7yjL8vG+3REAAAAAAAAAAAAAAAxPyhYNVJblVUneluRbSd7IgqWLjsWLqp4WMLo65eKPSTYry/JzZVnO6sNtAAAAAAAAAAAAAADAsKZs0WBlWb5WluUXk6yc5Lgkz2f+8kTZx49U1mtL8qsk25RluUNZlrf3680BAAAAAAAAAAAAAMAwUJRlrcMW6KuiKEYm2T7JPkl2SLJajenV/xhdnWwxNclfkvwmySVlWb7QiJwwlC0yepJvdAAAAAAAA9ictrZWRwAAAAAAoAdmz3yyq59zH3KULZqsKIrlk2yUZN3MLV4sl2SZJGOSjM7c00beSDIjyUtJnk7yeJK7k/wzyZ1lWfoXB6iDsgUAAAAAwMCmbAEAAAAAMDgoWwAMIcoWAAAAAAADm7IFAAAAAMDgMJzKFiNaHQAAAAAAAAAAAAAAAGAgUbYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoGNXqAMNZURQjkoxNsmiSRZIU7WNlWT7WqlwAAAAAAAAAAAAAADCcKVs0SVEU6yXZNsnGSdZLskKSZdL56SJl/LcBAAAAAAAAAAAAAICW8AP9/agoivWTHJZkvyTLVof6YZ/Nuxi+syzLmxu5HwAAAAAAAAAAAAAADGXKFv2gKIotkhyb5D3tlzqZVnb15b3Y8vUkP0znp2TckWSTXqwJAAAAAAAAAAAAAADDUmc/nE8vFUWxWFEUP0lyfeYWLYp5H2UnH//6sspHr5Rl+XCS8zus1f6xYVEUG/R2bQAAAAAAAAAAAAAAGG6ULRqkKIpNM/cUiUOyYMki6bwI0euCRSe+N+9zZ4WODzdwHwAAAAAAAAAAAAAAGNKULRqgKIo9k/wlyaTMX7LoWKqodcJFn5RleWuSv2b+Akd7hg8WRdHIYgcAAAAAAAAAAAAAAAxZyhZ9VBTFbkl+nmR0FixZJAsWK/rrZIskObsarfJ66SSbNXgvAAAAAAAAAAAAAAAYkka1OsBgVhTF2knOT7JQ5i9TtOt47cnMPQHj0SQvJlk/ycF5s6DRVxckOSnJyCx4asZ7ktzcgD0AAAAAAAAAAAAAAGBIc7JFLxVFMSrJRUnGZ8FSRfWEi6lJvp1kjbIsJ5VleVBZlseUZXlCkmsbmaksy5eTXJ/OixvvaeReAAAAAAAAAAAAAAAwVClb9N7RSdZN50WL9vc/TLJiWZZfKMvyoSbluqLD+/bSx5ZFUSzSpAwAAAAAAAAAAAAAADBoKVv0QlEUEzO3bFEtVlRPs3gjyYFlWR5RluWUJsf7S+V19YSLRZKs3+QsAAAAAAAAAAAAAAAw6Chb9M7hSSbMe91etGh/3ZbkQ2VZXtiKYEluTTJr3uuyw9jbmpwFAAAAAAAAAAAAAAAGHWWL3vlwFiwytJcuvlqW5S+aH2musizfSPJQF8PKFgAAAAAAAAAAAAAA0A1lizoVRbFOkrXb32b+0sWjSb7d9FALui9zs3WkbAEAAAAAAAAAAAAAAN1QtqjfNp1cay9dHFuW5awm5+nME51cK5JManYQAAAAAAAAAAAAAAAYbJQt6rdF5XX1VIvZSX7R5CxdeabD+/acE5odBAAAAAAAAAAAAAAABhtli/qt3uF9+6kWfy3LcmoL8nRmShfXxzc1BQAAAAAAAAAAAAAADELKFvVbKfOfaNHu9ibnqOWNLq472QIAAAAAAAAAAAAAALqhbFG/rgoLzzc1RW0LdXF90aamAAAAAAAAAAAAAACAQUjZon5jurj+QlNT1LZ4F9e7OvECAAAAAAAAAAAAAACYR9mifrO6uN5VCaMVuipbTG9qCgAAAAAAAAAAAAAAGISULer3ehfXl2hqitqW6eL6i01NAQAAAAAAAAAAAAAAg5CyRf1e7uL60k1NUdsWScrK+2Le+8dbEwcAAAAAAAAAAAAAAAYPZYv6PZK55YWqIsnbW5BlAUVRrJBk5fa3HYYnNzUMAAAAAAAAAAAAAAAMQsoW9Xuow/v2EyTWLYriLc0O04nta4zd1rQUAAAAAAAAAAAAAAAwSClb1O9vlddFh9e7NTlLZ46sMXZT01IAAAAAAAAAAAAAAMAgpWxRv+u7uF4kOaqZQRYIUBTbJnl75p62UeTNUzeS5OUkt7YiFwAAAAAAAAAAAAAADCbKFnUqy/LBJPe2v838pYYNiqJoyekWRVGMSHJcZ0OZm++3ZVm2NTcVAAAAAAAAAAAAAAAMPsoWvXNR5pYYqtqLFz8qimLJ5kfKcUm2ruTo6LzmxgEAAAAAAAAAAAAAgMFJ2aJ3Tk8yc97rjuWG5ZKcXRTFqGaFKYpinySfz5snbKTD6wfKsryqWXkAAAAAAAAAAAAAAGAwU7bohbIsn0xyTuYvWRR5s3ixU5LLi6IY399ZiqL4zyQXdMjRMdMJ/Z0DAAAAAAAAAAAAAACGCmWL3vtKkqnzXrefIlEtXOyQ5LqiKLboj82LoliuKIqzk3wnycjK/u152rPcleSM/sgAAAAAAAAAAAAAAABDkbJFL5Vl+VTmFi6KDkPVwsX6Sa4viuK8oig2asS+RVEsWxTFCUkeTHJQZb9/Rau8bktyZFmW1WsAAAAAAAAAAAAAAEANo1odYDAry/Kkoii2T7Jn3ixYJPMXLook+yfZvyiKB5NckuSWJHcnWai7PYqieEuSNZLsmGSPJJtX1k3mP1Ujlddlkv8py/K63t4fAAAAAAAAAAAAAAAMR4VDD/qmKIoJSa5Psm7mL1wknRchOv6BdxxrL0rckWSVJBO6mN/d2pcleV9Zlm3d3wUMbYuMnuQbHQAAAADAADanzT9nAAAAAAAMBrNnPll0P2toGNHqAINdWZZTkuyQ5N68WZSoFiGq16qnXVRPp0hlfvvnjZJM7GR+x/X/FaVy7YYkH1C0AAAAAAAAAAAAAACA+ilbNEBZls8leVeSv2TBkyeS+YsVZYePmkt3Mb9jUaN6/YokO5dlOaPuGwEAAAAAAAAAAAAAAJQtGqUsyxcz94SLk/Jm+aFjoaLjKRW1jlDpWKzo7Guqp2UkyXeSvLcsy2m9vhEAAAAAAAAAAAAAABjmlC0aqCzL2WVZfibJtknuTuenWfRUV2WMzk65uD/Je8qy/HxZlnN6kx0AAAAAAAAAAAAAAJhL2aIflGV5fZINknwoyV2ZvzhR1viYb5kac9rXezzJp5KsV5blH/vpdgAAAAAAAAAAAAAAYFgpyrKewxbojaIotkhySJLdkkzqMNzdf4COp1u8nuR3Sc5N8uuyLNsaEhKGsEVGT/KNDgAAAABgAJvT5p87AAAAAAAGg9kzn+z48+1DlrJFkxVFsU6SdyTZOMnbMrd8sXyScZm/WDEnyYtJHkvySJLbkvwtyQ1lWb7RzMww2ClbAAAAAAAMbMoWAAAAAACDg7IFLVEUxSJJFkryRlmWs1qdB4YKZQsAAAAAgIFN2QIAAAAAYHAYTmWLUa0OwJvmnVjh1AoAAAAAAAAAAAAAAGghZYs6FUWxeJJxXQzPLMvymWbmAQAAAAAAAAAAAAAAGmtEqwMMQj9L8kgXH//RulgAAAAAAAAAAAAAAEAjONmifqsmKTq53pbkpCZnAQAAAAAAAAAAAAAAGkzZon7LJik7uX5rWZZPNzsMAAAAAAAAAAAAAADQWCNaHWAQGtfhffspF39vdhAAAAAAAAAAAAAAAKDxlC3qN7OL6w83NQUAAAAAAAAAAAAAANAvlC3q91oX16c0NQUAAAAAAAAAAAAAANAvlC3q90IX1/1ZAgAAAAAAAAAAAADAEKAgUL/7kxSdXJ/Y7CAAAAAAAAAAAAAAAEDjKVvU794urq/S1BQAAAAAAAAAAAAAAEC/ULao3x87uVYk2azZQQAAAAAAAAAAAAAAgMZTtqjftUmmV96X8z5vUBTF0i3IAwAAAAAAAAAAAAAANJCyRZ3KsnwjycWZe5pF1YgkH25+IgAAAAAAAAAAAAAAoJGULXrnpA7vy8wtX3y2KIqxLcgDAAAAAAAAAAAAAAA0iLJFL5RleVuSC7Pg6RZLJ/lu8xMBAAAAAAAAAAAAAACNomzRe59J8lLlffvpFh8riuLjrYkEAAAAAAAAAAAAAAD0lbJFL5Vl+UyS/ZLMrl7O3MLF94ui+HJRFB1PvgAAAAAAAAAAAAAAAAY4ZYs+KMvyj0k+ks4LF8cm+WtRFOu3IBoAAAAAAAAAAAAAANBLyhZ9VJbl+Un2SDIlc0sWyZuFi82T3FIUxcVFUexeFIU/bwAAAAAAAAAAAAAAGOCKsixbnWFIKIpi+SQ/TrJL5pYtkvnLF0nyXJIbk/w9ya1JHkvyapIpZVlOaV5aGF4WGT3JNzoAAAAAgAFsTltbqyMAAAAAANADs2c+WXQ/a2hQtuiFoijm1BquvO5Yuqhea7ayLMtRLdobWkrZAgAAAABgYFO2AAAAAAAYHIZT2cIP3/dOT/8HUmRuuaLscA0AAAAAAAAAAAAAABiglC16r6vflN+xTNHxVItW/IZ9BQ8AAAAAAAAAAAAAAOghZYu+qbfE0IrSQyvKHQAAAAAAAAAAAAAAMGiNaHUAAAAAAAAAAAAAAACAgcTJFn3j1AgAAAAAAAAAAAAAABhilC16r2h1AAAAAAAAAAAAAAAAoPGULXrn2FYHAAAAAAAAAAAAAAAA+kdRlmWrMwD0q0VGT/KNDgAAAABgAJvT1tbqCAAAAAAA9MDsmU8Wrc7QLCNaHQAAAAAAAAAAAAAAAGAgUbYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoULYAAAAAAAAAAAAAAACoGNXqAMNVURQLJVk+yXJJlkgyOskiSdqSzEjyepLnkjxdluVzrcoJAAAAAAAAAAAAAADDjbJFkxRFsXGS7ZJslWS9JKulhyeLFEXxepK7k9ye5M9J/liW5TP9kxQAAAAAAAAAAAAAAIa3oizLVmcYsoqiWC/JR5N8IHNPsfjXUC+XLCuf/5rkvCRnl2U5tdchYRhYZPQk3+gAAAAAAAawOW1trY4AAAAAAEAPzJ75ZG9/Fn7QUbboB0VRvDPJfyV5T/ulDlN684fe1RpTk5ye5FtlWb7Yi3VhyFO2AAAAAAAY2JQtAAAAAAAGh+FUthjR6gBDSVEUyxVFcVGSP2du0aKY91F2+PjXl/TwI52s0T42Icl/JnmgKIpP9+PtAQAAAAAAAAAAAADAsKBs0SBFUeyT5M4k+2TBkkXSdYmiR8t38nUdixeLJfluURTXFEWxQl/uBQAAAAAAAAAAAAAAhjNliwYoiuLYJBclWTzzlyy6KlZ0PKWiJx/zbZnOixdFkncnuakois0adX8AAAAAAAAAAAAAADCcKFv0UVEUpyb5cjovWVR1Vpzo7LSLrk7A6Kp80XFOkiyb5E9FUbyzb3cHAAAAAAAAAAAAAADDz6hWBxjMiqI4Lsnh8962Fx06K1mkw9icJA8kuSPJ00mmVD4WSjKh8rFmkg2TLN1hzY77dSxcjEny26IotinL8h/13hsAAAAAAAAAAAAAAAxXRVmW3c9iAUVR7JHk0nRfsmi//niSC5L8IskdZVnOqHO/ZZJsneTAJLsnGZ3Oixwd934gyaZlWb5Wz34wlCwyepJvdAAAAAAAA9ictrZWRwAAAAAAoAdmz3yy48/ND1nKFr1QFMXEzC0xLJm5xYZaRYsbk3yxLMs/N3D/8Uk+kuQrSZZI54WP9lxlkh+WZXlko/aHwUbZAgAAAABgYFO2AAAAAAAYHIZT2WJEqwMMUl9J50WLsnLt6SQfLMty60YWLZKkLMupZVmenGSNJCfnzbJFxx8ob8/yb0VRrNfIDAAAAAAAAAAAAAAAMFQpW9Rp3qkW/56uiw3tp1lsUpbl+f2ZpSzLV8qy/HSSPZO8XsmRzF8CGZHkc/2ZBQAAAAAAAAAAAAAAhgpli/odnGTMvNfthYb2okWZ5IYkO5Vl+VyzApVleUWS3ZK8UcmTyusiyQeKolisWZkAAAAAAAAAAAAAAGCwUrao354d3leLDc8l2acsy9fTZGVZXpfkU5n/RIvq64WT7NzUUAAAAAAAAAAAAAAAMAgpW9ShKIqRSbbJ/AWL5M1TLT5fluWzTQ82T1mWP05yXSVPR9s1NxEAAAAAAAAAAAAAAAw+yhb1WT3JIvNedyw0PJbk7KYnWtA3a4yt17QUAAAAAAAAAAAAAAAwSClb1GfVTq61ly7OK8uys9Mkmqosy98laT9do6x8LtJ5fgAAAAAAAAAAAAAAoELZoj4Ta4xd07QU3ftT5pYrOprQ5BwAAAAAAAAAAAAAADDoKFvUZ+EaY/c0LUX37u7ieq38AAAAAAAAAAAAAABAlC3q9VqNsRealqJ7L3ZxvVZ+AAAAAAAAAAAAAAAgyhb1erXG2JympeheV1lq5QcAAAAAAAAAAAAAAKJsUa+HaoxNbFqK7nXMUiQpkzzcgiwAAAAAAAAAAAAAADCoKFvUoSzLyUleb3/bYXiN5qapqassdzU1BQAAAAAAAAAAAAAADELKFvW7LnNPiujonc0OUsPWWbAMkiTXNjsIAAAAAAAAAAAAAAAMNsoW9bu0i+v7NjVFF4qiWDfJ2zoZmpXkyibHAQAAAAAAAAAAAACAQUfZon4XJplWeV9m7kkXby+KYsvWRJrPZzu8LzI34y/KsnytBXkAAAAAAAAAAAAAAGBQUbaoU1mWLyU5I3NLDFVFku8XRTGy+anmBSiKdyT5cOaWKzr6dpPjAAAAAAAAAAAAAADAoKRs0TtfT/Jc5X17uWGTJN9repokRVEsneSCzP/ftP1Ui5+WZXl7K3IBAAAAAAAAAAAAAMBgo2zRC2VZvpjkyMx/ukU57/2RRVF8p5l5iqJYLsnVSVbKgqdaPJ7kM83MAwAAAAAAAAAAAAAAg5myRS+VZfmLJF9M54WL/yyK4rdFUazQ3zmKotg9yd+TrJv5ixZFkpeS7FKW5ZT+zgEAAAAAAAAAAAAAAEOFskUflGX5rSRf7Xg5c4sOuya5tyiKrxVFsWij9y6KYp2iKK5McmmSZTN/6aNI8kySncuyvLfRewMAAAAAAAAAAAAAwFBWlGXZ/SxqKopi/yQ/TDIhb54u0V5+KJNMSfLLJOcn+UNZlm293GfxJPslOTDJO+ftUXSy581J9inL8one7ANDzSKjJ/lGBwAAAAAwgM1p69U/nQAAAAAA0GSzZz5ZdD9raFC26IWiKFbs5PKkJCcn2Shvlh+S+UsXSTI1yT+S3DHv81OZW8aYMm9socwtbYxPMjHJGkk2nPexZpKRXaxbJGlLcmqS/00yp5e3V1NZlo/1x7rQn5QtAAAAAAAGNmULAAAAAIDBQdmCmoqiaMv8hYr5hiuvOytddLze4207vO94mkVv161HWZblqH7eAxpO2QIAAAAAYGBTtgAAAAAAGByGU9nCD873Xk/+R1I9faKr4kU9Ov7AeMd1hs3/cAEAAAAAAAAAAAAAoL8oW/ReT0626Oxax+JFvWoVKvrzt/crcgAAAAAAAAAAAAAAMCwoW/RNbwoI/Vla6K+1+7PEAQAAAAAAAAAAAAAAA8qIVgcAAAAAAAAAAAAAAAAYSJxs0TdOfAAAAAAAAAAAAAAAgCFG2aL3ilYHAAAAAAAAAAAAAAAAGk/Zonc+2uoAAAAAAAAAAAAAAABA/yjKsmx1BoB+tcjoSb7RAQAAAAAMYHPa2lodAQAAAACAHpg988mi1RmaZUSrAwAAAAAAAAAAAAAAAAwkyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVyhYAAAAAAAAAAAAAAAAVo1odAAAAAAAAAAAAYDBbeOGFs+aaq2aFty6XcePHZcyii2ba9Ol5bepreeLJp3PffQ9l1qxZrY4JAADUQdkCAACAQWfllSdl/fXWzmqrrZxJk5bPCissn0mT3prFF5+YMWPGZMyYRTNmzKKZPXt2Zsx4I6+8OiXPPvN8Hnvsidxz7wO5/bZ/5rrrb8orr7za6ltJkowaNSprrbV61l13rayzzppZd5218ta3LpfFFpuQiRMnZOLE8ZkzZ06mT5+RV155NU89/WwmT348/7zzntzy9zty441/H3D/SDcU7wkAAAAAqM+oUaPytrfNfU647jprZd115z0nnDghiy02cb7nhC+//GqefvrZPDL58dx559255ZY7csONtwzo54TvePsm2WuvnbPzzttn3XXWzKhRXf8o1uzZs3PX3ffnyiv/kEsvvSp/u+nWJiYFAAB6oyjLstUZAPrVIqMn+UYHADCIrbTSCtlyy82y1ZabZ8MN1826666V8ePH9Xndtra2/O1vt+aSS36bc869JC+//Erfw/ZQURTZaKP1st27t8p2270zW221ecaNG9vr9V5/fVquvvranHPOz3PZ5Vdnzpw5DUzbM0PxngAAaJ45bW2tjgAAQAMURZGNN1ov2223dbbf7p3Zeuu39/k54e+v/nN+dvbFueyygfOccL/93pujPntENt1kg16vccvf78iJ3/2/XHzxpQ1M1nerrbZyNttsw2y2yYbZbLMNs9FG69V8Jj958uNZfc0tmpgQAIBWmz3zyaLVGZpF2QIY8pQtAAAGp/8+/kv5wAf2ygorLNfve02bNj1nnXVhvnn89/L88y/2yx4jR47M9tu9M/vsu0feu+fOWWKJt/TLPo888mi+c8KpOfPMC9LWzz+wNhTvCQCA1lC2AAAYvEaOHJkdtn9n9t13z+z13l367Tnhww8/mm9/55T85MzzW/accK21VsupP/hWtt12q4at+ac/3ZAjP/GF3H//Qw1bs6cmTVo+m266YTbfbMNsusmG2XTTDfKWtyxW1xrKFgAAw4+yBcAQomwBADA43XD9Zdl0097/VrDemDJlao754jfz4x+f27A11157zXzyk4dlr/fukiWXXLxh63bn1lv/kcOPODp33HFXw9ceivcEAEBrKVsAAAw+66yzZj71yY/lfXvt2tTnhH+/9R/598OPyu23N/c54fvet2vOPON7DTl5uaOpU1/LRw79dH796ysbvna7pZdeMptvtlE2m1es2GyzDbP00kv2eV1lCwCA4Wc4lS1GtDoAAAAADBQTJozPKT/4Vi44/4cZPXp0Q9bcfff35LBDD2rqPzYmySabbJBr//yrfOxjH2z42kPxngAAAACA+uyx+4752GEfbPpzwk032SDXXXtp/u1jH2rankccfkguuuBH/VK0SJLx48fl4gtPz+H/fki/rJ8kl192Xn79q7PyX1/+z+y22w4NKVoAAMBQp2wBAAAAHbz//bvlyivOz9ixY1odpU9Gjx6dU37wrXzlK59tdZSGGYr3BAAAAADUZ/To0Tnt1P/J1756VL/vdfDB++Wk7x2XESP698esRowYke+fdFw+9KF9+3UfAACg50a1OgAAAAD01pw5c/L440/mgQceycMPP5pX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Bx50RMueE3ZXtthww3UzYsSItLW1NXzvkSNHZoMN1qk5R9kCAACaT9mC4eK1JGMz9ySIosPna5N8rWXJ+mb3JEflzfsBAAD6aOyYMTXHp70+veF7brPNFrnootMzenTtIscf/3hdPrD//8usWbManqHRhuI9AQAAAAD12XabLfPzi8/o9jnhNddcl30/8G8tfU54y9//kenTp2fRRRftdHz8+HHZdJMNcvMttzd877dvvnHGju362fT06dPz91trl0EAAIDGU7ZguLgtybsyt5TQrr2gsFxZln9uSao+Kopi9VZnAACAoWaFFZavOf7yK682dL93vGOT/OKSMzNmTOf/gNfuuutvyt77HJo33nijofv3h6F4TwAAAABAfbZ4x6b51S9/2v1zwuv+lvft/ZGWPyd84403csMNt2SHHd7V5Zz3vGebfilb7LDDO2uOX3fdTS3/8wEAgOFoRKsDQJPcXGNs9aIoxjctCQAAMGAtv/yyWXLJxWvOeeSRRxu230YbrZdLf/2zjB8/rua8m2++Pe973yGZNq3xp2o02lC8JwAAAACgPhtvtF5++5uze/Cc8LbsudeHB8xzwqv/cG3N8fe9b9d+2XfvvXevOf77qwfl7xAFAIBBT9mC4eKWDu+LDq83bWIWAABggHrPe7apOT5lytQ88cTTDdlrnXXWymW/PTeLLTax5rw77rgre+z5oUyd+lpD9u1PQ/GeAAAAAID6rLvuWrni8vO7fU54+x13ZdfdPzignhNe8ovLao5vuskGWXPN1Rq65zrrrJkN1l+n5pxf/PLyhu4JAAD0jLIFw0Wtky2SZLOmpAAAAAa0gw/er+b4DTd091eLnllj9VVyxeXndXuKxt1335fddj8or7zyakP27U9D8Z4AAAAAgPqsscaq+d0VF3T7nPCuu+/LLrseMOCeEz788KP561//XnPOx4/8aEP3/MTHD6s5fv31N2Xy5McbuicAANAzyhYMC2VZPpTklfa3nUxRtgAAgGFu2223zDbv2qLmnN//vu9Hta+00gq54soLsuyyS9ec98ADD2fX3Q7KCy+81Oc9+9tQvCcAAAAAoD4rrbRCrrrywm6fE97/wMPZeZcDBuxzwjN/ekHN8Y8csn+399hTb33rcjn4Q/vUnHPWzy5qyF4AAED9lC0YTm5JUnS4Vs67tnnz4wAAAAPFuHFjc9qp3645Z9asWbnwol/3aZ/lllsmV15xQSatsHzNeZMnP5Zddj0gzzzzXJ/2a4aheE8AAAAAQH2WW26ZXHXlhZk0qfZzwkceeSw77bz/gH5OeM65l+TZZ5/vcnzs2DE5/ptfbMhe/338F7Pooot2Of7MM8/lnHMvacheAABA/ZQtGE5u6fC+WrxYuSiKtzQzDAAA0LltttkiEyaMb9p+iy46OhdfdHpWW23lmvMuuvjSPP/8i73eZ8klF88Vl5+XVVddqea8x594KjvvckCeeOLpXu/VLEPxngAAAACA+iy55OL53ZUXdPuM9fHHn8qOO38gTzzxVHOC9dIbb7yRk39wRs05Hz54v+y11y592mefffbIQQfuXXPO90/+cWbOnNmnfQAAgN5TtmA4ubmb8c2akgIAAKjp4IM/kPvuvSFf+MKnMm7c2H7da43VV8lVv7so22//rprzZs6cmeOO+99e7zNx4oRc9ttzs/baa9ac9/TTz2bXXQ7I5MmP93qvZhmK9wQAAAAA1GfixAm54vLzs04PnhPutMv+g+Y54UnfPz2PPvpEzTlnnvG9bL7ZRr1a/x1v3yQ//tGJNec8+ugT+f7JtUsfAABA/1K2YDjpeLJFR8oWAAAwQCy++GI59mufy/333ZgTT/hattyysf93fezYMfnqV4/K3//++7z97Rt3O/+bx5+Uhx9+tNd7/fpXZ2WjjdarOe/551/MrrsdlAcefKRX+zTTULwnAAAAAKA+Y8eOyW8vPTsb9+A54c67HpAHHni4Scn6bvr0Gfnc579ec86ECeNzxeXnZffd3lPX2nvuuVMuv+zcjB8/rua8o44+NjNmzKhrbQAAoLGKsixbnQGapiiKZ5Ms2f42SVn5/KuyLPdpVbbeKIrisCSnZ/77KJKUZVmObGW2gWSR0ZN8owMAGEROP/27+fDB+y1w/Yknns4vf3lZ/vCHv+RvN92al156pa51x40bm6232jwHHPj+7PXeXTJ27Jgefd0f/3R9dtvtoLS1tdW1X7tfXPKT7L77jt3OO+20n+aOf9zVqz1645mnn8sVV17Tq68divcEAEBrzenl/98GAKB1fvXLn2aPHjwnPOXUM3PHHU18TvjMc7n8ij80ZK2fnXVyDjpw75pz2tracsGFv8o3j/9e7rvvoS7nrb32Gvnylz6T/T+wV7f7nnveJTnkI5+qO2933vXOd2SNNVat62vWWnO1fPazR3Q5/sILL+WLXzq+7izX/uWvedAv6gEAGJRmz3yyaHWGZlG2YFgpiuK3SXbL/OWEzHv9RFmWK7YqW290KFv863KULeajbAEAMLh0Vbbo6PEnnsoD9z+UyY8+kWeffT4vvfhy3nhjZmbPmZ3x48Zl3PixmTB+fCZNWj4bbLBOVl11pYwYUd8Bj3feeU+232GfTJkytbe3k/vuuyErrzSp11/fX/587Y3ZaacP9Oprh+I9AQDQWsoWAACDz4P3/zUrrzwAnxP++YbssGP3z5h7YuzYMfnrjZdn7bet0aP5t952Z2688ZZMnvx4Xnvt9YwfPzYrr7xittpq82y04bo9WuOeex/IFlvultdfn9aX6J0648f/m0M+PDCeoR562Gfys7MvanUMAAB6YTiVLUa1OgA02S2ZW7ZI5i8oJMlbi6JYqizL55ucqRGGzTctAABoN2mF5TNpheX7bf3rb7gpe+99aJ+KFgAAAAAADF6vvz4tu+1+UP50zS+z0kordDt/k43XzyYbr9/r/R599InstvtB/VK0AAAA6qdswXBzc+V1e0GhWrrYPMnlzYvTZ/cnOavVIQAAYChpa2vLqaeemWO+eHxmzpzZ6jgAAAAAALTQ448/lZ122T+X/eacrL76Kv22zwMPPJLd9/xgHn/8qX7bAwAAqI+yBcPNzUn+kQVPtWi3TBOz9FlZln9J8pdW5wAAgKHijjvuyueOPjZ//vONrY4CAAAAAMAA8dBDk7PFVrvn3LNPyc47b9fw9a+88pp86MOfyCuvvNrwtQEAgN5TtmBYKcvyuSQbtToHAADQtR/84Iy88PyL2Xnn7bLuums1Zc+bbrotJ//gjFx88aUpy6662QAAAAAADFevvPJqdt/zQzn44P3yreO/lGWWWarPaz777PP5/DHH5Zxzft6AhAAAQKMpWwAAADCg3HHHXbnjjrtyzBe/mUmTls9OO707W2yxWd6++UZZc83VMmLEiD7vMWfOnPzzn/fmt5f9Pr/4xWX55z/vbUByAAAAAACGurPPvjiXXPLbfPjgD+TIIz+SddZes+417rr7vpx22lk562cXZvr0Gf2QEgAAaITCb+wEhrpFRk/yjQ4AYIiYMGF8Ntlk/ay5xmpZeeVJWXnlSVlppUlZfPHFMmbMohk3bmwWXXR05syZkzfemJlp06bnhRdeynPPPZ/Jjz6R++97MHfddV/++rdb8+qrU1p9OwAAwDxz2tpaHQEAAHpljTVWzc47vTsbb7x+1llnzbx1+WUzfvy4jBmzaKZNm56pU1/Lk08+nbvveSC33XZnrvzdH/Pgg4+0OjYAAPTa7JlPFq3O0CzKFsCQp2wBAAAAADCwKVsAAAAAAAwOw6lsMaLVAQAAAAAAAAAAAAAAAAYSZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAAAAAAAIAKZQsAAAAAAAAA4P+zd+9Btl5lncd/KxcCIQkockdIUEIgaEIk8cIICQMCDpaIXAQHCDOCjKBVolLOOMPFmT+ckRlQBxQpJVHwMkopCoIgEFCQEggJDCAEDARRCATNlYQkPPPHbsfHkN5vd59+9z59+vOp6kpV1tprPed071OVk/72CwAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAAAADQiC0AAAAAAAAAAAAAAAAasQUAAAAAAAAAAAAAAEAjtgAAAAAAAAAAAAAAAGjEFgAAAAAAAAAAAAAAAI3YAgAAAAAAAAAAAAAAoBFbAAAAAAAAAAAAAAAANGILAAAAAAAAAAAAAACARmwBAAAAAAAAAAAA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ACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACPCFgAAAAAAAAAAAAAAACP/B5SJ6HjpUoD2AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 3600x2400 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "{0.0: 9309, 1.0: 523}"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "plt.figure(dpi=600)\n",
    "mat = confusion_matrix(y_test, predicted_naive)\n",
    "sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)\n",
    "\n",
    "plt.title('Confusion Matrix for Naive Bayes')\n",
    "plt.xlabel('true label')\n",
    "plt.ylabel('predicted label')\n",
    "plt.savefig(\"confusion_matrix.png\")\n",
    "plt.show()\n",
    "\n",
    "type(predicted_naive)\n",
    "unique, counts = np.unique(predicted_naive, return_counts=True)\n",
    "dict(zip(unique, counts))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "id": "57d456bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy with Naive-bayes:  0.9403986981285598\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "score_naive = accuracy_score(predicted_naive, y_test)\n",
    "print(\"Accuracy with Naive-bayes: \",score_naive)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
