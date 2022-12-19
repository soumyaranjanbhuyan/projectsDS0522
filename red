{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "456eb004",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import sklearn\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "91dfa7f8",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.700</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.99780</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.880</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.098</td>\n",
       "      <td>25.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>0.99680</td>\n",
       "      <td>3.20</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.760</td>\n",
       "      <td>0.04</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.092</td>\n",
       "      <td>15.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0.99700</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.65</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.2</td>\n",
       "      <td>0.280</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.075</td>\n",
       "      <td>17.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>0.99800</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.58</td>\n",
       "      <td>9.8</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.700</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.99780</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1594</th>\n",
       "      <td>6.2</td>\n",
       "      <td>0.600</td>\n",
       "      <td>0.08</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.090</td>\n",
       "      <td>32.0</td>\n",
       "      <td>44.0</td>\n",
       "      <td>0.99490</td>\n",
       "      <td>3.45</td>\n",
       "      <td>0.58</td>\n",
       "      <td>10.5</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1595</th>\n",
       "      <td>5.9</td>\n",
       "      <td>0.550</td>\n",
       "      <td>0.10</td>\n",
       "      <td>2.2</td>\n",
       "      <td>0.062</td>\n",
       "      <td>39.0</td>\n",
       "      <td>51.0</td>\n",
       "      <td>0.99512</td>\n",
       "      <td>3.52</td>\n",
       "      <td>0.76</td>\n",
       "      <td>11.2</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1596</th>\n",
       "      <td>6.3</td>\n",
       "      <td>0.510</td>\n",
       "      <td>0.13</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.076</td>\n",
       "      <td>29.0</td>\n",
       "      <td>40.0</td>\n",
       "      <td>0.99574</td>\n",
       "      <td>3.42</td>\n",
       "      <td>0.75</td>\n",
       "      <td>11.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1597</th>\n",
       "      <td>5.9</td>\n",
       "      <td>0.645</td>\n",
       "      <td>0.12</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.075</td>\n",
       "      <td>32.0</td>\n",
       "      <td>44.0</td>\n",
       "      <td>0.99547</td>\n",
       "      <td>3.57</td>\n",
       "      <td>0.71</td>\n",
       "      <td>10.2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1598</th>\n",
       "      <td>6.0</td>\n",
       "      <td>0.310</td>\n",
       "      <td>0.47</td>\n",
       "      <td>3.6</td>\n",
       "      <td>0.067</td>\n",
       "      <td>18.0</td>\n",
       "      <td>42.0</td>\n",
       "      <td>0.99549</td>\n",
       "      <td>3.39</td>\n",
       "      <td>0.66</td>\n",
       "      <td>11.0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1599 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "0               7.4             0.700         0.00             1.9      0.076   \n",
       "1               7.8             0.880         0.00             2.6      0.098   \n",
       "2               7.8             0.760         0.04             2.3      0.092   \n",
       "3              11.2             0.280         0.56             1.9      0.075   \n",
       "4               7.4             0.700         0.00             1.9      0.076   \n",
       "...             ...               ...          ...             ...        ...   \n",
       "1594            6.2             0.600         0.08             2.0      0.090   \n",
       "1595            5.9             0.550         0.10             2.2      0.062   \n",
       "1596            6.3             0.510         0.13             2.3      0.076   \n",
       "1597            5.9             0.645         0.12             2.0      0.075   \n",
       "1598            6.0             0.310         0.47             3.6      0.067   \n",
       "\n",
       "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "0                    11.0                  34.0  0.99780  3.51       0.56   \n",
       "1                    25.0                  67.0  0.99680  3.20       0.68   \n",
       "2                    15.0                  54.0  0.99700  3.26       0.65   \n",
       "3                    17.0                  60.0  0.99800  3.16       0.58   \n",
       "4                    11.0                  34.0  0.99780  3.51       0.56   \n",
       "...                   ...                   ...      ...   ...        ...   \n",
       "1594                 32.0                  44.0  0.99490  3.45       0.58   \n",
       "1595                 39.0                  51.0  0.99512  3.52       0.76   \n",
       "1596                 29.0                  40.0  0.99574  3.42       0.75   \n",
       "1597                 32.0                  44.0  0.99547  3.57       0.71   \n",
       "1598                 18.0                  42.0  0.99549  3.39       0.66   \n",
       "\n",
       "      alcohol  quality  \n",
       "0         9.4        5  \n",
       "1         9.8        5  \n",
       "2         9.8        5  \n",
       "3         9.8        6  \n",
       "4         9.4        5  \n",
       "...       ...      ...  \n",
       "1594     10.5        5  \n",
       "1595     11.2        6  \n",
       "1596     11.0        6  \n",
       "1597     10.2        5  \n",
       "1598     11.0        6  \n",
       "\n",
       "[1599 rows x 12 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data= pd.read_csv(\"red.csv\")\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "122ef362",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.88</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.098</td>\n",
       "      <td>25.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>0.9968</td>\n",
       "      <td>3.20</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.76</td>\n",
       "      <td>0.04</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.092</td>\n",
       "      <td>15.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0.9970</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.65</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.2</td>\n",
       "      <td>0.28</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.075</td>\n",
       "      <td>17.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>0.9980</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.58</td>\n",
       "      <td>9.8</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "0            7.4              0.70         0.00             1.9      0.076   \n",
       "1            7.8              0.88         0.00             2.6      0.098   \n",
       "2            7.8              0.76         0.04             2.3      0.092   \n",
       "3           11.2              0.28         0.56             1.9      0.075   \n",
       "4            7.4              0.70         0.00             1.9      0.076   \n",
       "\n",
       "   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "0                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "1                 25.0                  67.0   0.9968  3.20       0.68   \n",
       "2                 15.0                  54.0   0.9970  3.26       0.65   \n",
       "3                 17.0                  60.0   0.9980  3.16       0.58   \n",
       "4                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "\n",
       "   alcohol  quality  \n",
       "0      9.4        5  \n",
       "1      9.8        5  \n",
       "2      9.8        5  \n",
       "3      9.8        6  \n",
       "4      9.4        5  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f5f73ec0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',\n",
       "       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',\n",
       "       'pH', 'sulphates', 'alcohol', 'quality'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "40477e61",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "      <td>1599.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>8.319637</td>\n",
       "      <td>0.527821</td>\n",
       "      <td>0.270976</td>\n",
       "      <td>2.538806</td>\n",
       "      <td>0.087467</td>\n",
       "      <td>15.874922</td>\n",
       "      <td>46.467792</td>\n",
       "      <td>0.996747</td>\n",
       "      <td>3.311113</td>\n",
       "      <td>0.658149</td>\n",
       "      <td>10.422983</td>\n",
       "      <td>5.636023</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.741096</td>\n",
       "      <td>0.179060</td>\n",
       "      <td>0.194801</td>\n",
       "      <td>1.409928</td>\n",
       "      <td>0.047065</td>\n",
       "      <td>10.460157</td>\n",
       "      <td>32.895324</td>\n",
       "      <td>0.001887</td>\n",
       "      <td>0.154386</td>\n",
       "      <td>0.169507</td>\n",
       "      <td>1.065668</td>\n",
       "      <td>0.807569</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4.600000</td>\n",
       "      <td>0.120000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>0.012000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0.990070</td>\n",
       "      <td>2.740000</td>\n",
       "      <td>0.330000</td>\n",
       "      <td>8.400000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>7.100000</td>\n",
       "      <td>0.390000</td>\n",
       "      <td>0.090000</td>\n",
       "      <td>1.900000</td>\n",
       "      <td>0.070000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.995600</td>\n",
       "      <td>3.210000</td>\n",
       "      <td>0.550000</td>\n",
       "      <td>9.500000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>7.900000</td>\n",
       "      <td>0.520000</td>\n",
       "      <td>0.260000</td>\n",
       "      <td>2.200000</td>\n",
       "      <td>0.079000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>0.996750</td>\n",
       "      <td>3.310000</td>\n",
       "      <td>0.620000</td>\n",
       "      <td>10.200000</td>\n",
       "      <td>6.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>0.640000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>2.600000</td>\n",
       "      <td>0.090000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>62.000000</td>\n",
       "      <td>0.997835</td>\n",
       "      <td>3.400000</td>\n",
       "      <td>0.730000</td>\n",
       "      <td>11.100000</td>\n",
       "      <td>6.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>15.900000</td>\n",
       "      <td>1.580000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>15.500000</td>\n",
       "      <td>0.611000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>289.000000</td>\n",
       "      <td>1.003690</td>\n",
       "      <td>4.010000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>14.900000</td>\n",
       "      <td>8.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       fixed acidity  volatile acidity  citric acid  residual sugar  \\\n",
       "count    1599.000000       1599.000000  1599.000000     1599.000000   \n",
       "mean        8.319637          0.527821     0.270976        2.538806   \n",
       "std         1.741096          0.179060     0.194801        1.409928   \n",
       "min         4.600000          0.120000     0.000000        0.900000   \n",
       "25%         7.100000          0.390000     0.090000        1.900000   \n",
       "50%         7.900000          0.520000     0.260000        2.200000   \n",
       "75%         9.200000          0.640000     0.420000        2.600000   \n",
       "max        15.900000          1.580000     1.000000       15.500000   \n",
       "\n",
       "         chlorides  free sulfur dioxide  total sulfur dioxide      density  \\\n",
       "count  1599.000000          1599.000000           1599.000000  1599.000000   \n",
       "mean      0.087467            15.874922             46.467792     0.996747   \n",
       "std       0.047065            10.460157             32.895324     0.001887   \n",
       "min       0.012000             1.000000              6.000000     0.990070   \n",
       "25%       0.070000             7.000000             22.000000     0.995600   \n",
       "50%       0.079000            14.000000             38.000000     0.996750   \n",
       "75%       0.090000            21.000000             62.000000     0.997835   \n",
       "max       0.611000            72.000000            289.000000     1.003690   \n",
       "\n",
       "                pH    sulphates      alcohol      quality  \n",
       "count  1599.000000  1599.000000  1599.000000  1599.000000  \n",
       "mean      3.311113     0.658149    10.422983     5.636023  \n",
       "std       0.154386     0.169507     1.065668     0.807569  \n",
       "min       2.740000     0.330000     8.400000     3.000000  \n",
       "25%       3.210000     0.550000     9.500000     5.000000  \n",
       "50%       3.310000     0.620000    10.200000     6.000000  \n",
       "75%       3.400000     0.730000    11.100000     6.000000  \n",
       "max       4.010000     2.000000    14.900000     8.000000  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e2a2a459",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x8bd4ca0>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.jointplot(x='residual sugar', y='quality', data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ef0133cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x8bd4a48>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.jointplot(x='free sulfur dioxide', y='quality', data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b88a3547",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x8bd4b38>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.jointplot(x='total sulfur dioxide', y='quality', data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b21e023a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "145.0"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "UpperLimit = np.percentile(data['total sulfur dioxide'], [99])[0]\n",
    "UpperLimit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1c84d40a",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>8.9</td>\n",
       "      <td>0.620</td>\n",
       "      <td>0.19</td>\n",
       "      <td>3.9</td>\n",
       "      <td>0.170</td>\n",
       "      <td>51.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>0.99860</td>\n",
       "      <td>3.17</td>\n",
       "      <td>0.93</td>\n",
       "      <td>9.2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>8.1</td>\n",
       "      <td>0.785</td>\n",
       "      <td>0.52</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.122</td>\n",
       "      <td>37.0</td>\n",
       "      <td>153.0</td>\n",
       "      <td>0.99690</td>\n",
       "      <td>3.21</td>\n",
       "      <td>0.69</td>\n",
       "      <td>9.3</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>354</th>\n",
       "      <td>6.1</td>\n",
       "      <td>0.210</td>\n",
       "      <td>0.40</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.066</td>\n",
       "      <td>40.5</td>\n",
       "      <td>165.0</td>\n",
       "      <td>0.99120</td>\n",
       "      <td>3.25</td>\n",
       "      <td>0.59</td>\n",
       "      <td>11.9</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>515</th>\n",
       "      <td>8.5</td>\n",
       "      <td>0.655</td>\n",
       "      <td>0.49</td>\n",
       "      <td>6.1</td>\n",
       "      <td>0.122</td>\n",
       "      <td>34.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>1.00100</td>\n",
       "      <td>3.31</td>\n",
       "      <td>1.14</td>\n",
       "      <td>9.3</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>591</th>\n",
       "      <td>6.6</td>\n",
       "      <td>0.390</td>\n",
       "      <td>0.49</td>\n",
       "      <td>1.7</td>\n",
       "      <td>0.070</td>\n",
       "      <td>23.0</td>\n",
       "      <td>149.0</td>\n",
       "      <td>0.99220</td>\n",
       "      <td>3.12</td>\n",
       "      <td>0.50</td>\n",
       "      <td>11.5</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>636</th>\n",
       "      <td>9.6</td>\n",
       "      <td>0.880</td>\n",
       "      <td>0.28</td>\n",
       "      <td>2.4</td>\n",
       "      <td>0.086</td>\n",
       "      <td>30.0</td>\n",
       "      <td>147.0</td>\n",
       "      <td>0.99790</td>\n",
       "      <td>3.24</td>\n",
       "      <td>0.53</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>649</th>\n",
       "      <td>6.7</td>\n",
       "      <td>0.420</td>\n",
       "      <td>0.27</td>\n",
       "      <td>8.6</td>\n",
       "      <td>0.068</td>\n",
       "      <td>24.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>0.99480</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.57</td>\n",
       "      <td>11.3</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>651</th>\n",
       "      <td>9.8</td>\n",
       "      <td>0.880</td>\n",
       "      <td>0.25</td>\n",
       "      <td>2.5</td>\n",
       "      <td>0.104</td>\n",
       "      <td>35.0</td>\n",
       "      <td>155.0</td>\n",
       "      <td>1.00100</td>\n",
       "      <td>3.41</td>\n",
       "      <td>0.67</td>\n",
       "      <td>11.2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>672</th>\n",
       "      <td>9.8</td>\n",
       "      <td>1.240</td>\n",
       "      <td>0.34</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.079</td>\n",
       "      <td>32.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>0.99800</td>\n",
       "      <td>3.15</td>\n",
       "      <td>0.53</td>\n",
       "      <td>9.5</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>684</th>\n",
       "      <td>9.8</td>\n",
       "      <td>0.980</td>\n",
       "      <td>0.32</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.078</td>\n",
       "      <td>35.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>0.99800</td>\n",
       "      <td>3.25</td>\n",
       "      <td>0.48</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1079</th>\n",
       "      <td>7.9</td>\n",
       "      <td>0.300</td>\n",
       "      <td>0.68</td>\n",
       "      <td>8.3</td>\n",
       "      <td>0.050</td>\n",
       "      <td>37.5</td>\n",
       "      <td>278.0</td>\n",
       "      <td>0.99316</td>\n",
       "      <td>3.01</td>\n",
       "      <td>0.51</td>\n",
       "      <td>12.3</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1081</th>\n",
       "      <td>7.9</td>\n",
       "      <td>0.300</td>\n",
       "      <td>0.68</td>\n",
       "      <td>8.3</td>\n",
       "      <td>0.050</td>\n",
       "      <td>37.5</td>\n",
       "      <td>289.0</td>\n",
       "      <td>0.99316</td>\n",
       "      <td>3.01</td>\n",
       "      <td>0.51</td>\n",
       "      <td>12.3</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1244</th>\n",
       "      <td>5.9</td>\n",
       "      <td>0.290</td>\n",
       "      <td>0.25</td>\n",
       "      <td>13.4</td>\n",
       "      <td>0.067</td>\n",
       "      <td>72.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>0.99721</td>\n",
       "      <td>3.33</td>\n",
       "      <td>0.54</td>\n",
       "      <td>10.3</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1493</th>\n",
       "      <td>7.7</td>\n",
       "      <td>0.540</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.089</td>\n",
       "      <td>23.0</td>\n",
       "      <td>147.0</td>\n",
       "      <td>0.99636</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.59</td>\n",
       "      <td>9.7</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1496</th>\n",
       "      <td>7.7</td>\n",
       "      <td>0.540</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.089</td>\n",
       "      <td>23.0</td>\n",
       "      <td>147.0</td>\n",
       "      <td>0.99636</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.59</td>\n",
       "      <td>9.7</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "15              8.9             0.620         0.19             3.9      0.170   \n",
       "109             8.1             0.785         0.52             2.0      0.122   \n",
       "354             6.1             0.210         0.40             1.4      0.066   \n",
       "515             8.5             0.655         0.49             6.1      0.122   \n",
       "591             6.6             0.390         0.49             1.7      0.070   \n",
       "636             9.6             0.880         0.28             2.4      0.086   \n",
       "649             6.7             0.420         0.27             8.6      0.068   \n",
       "651             9.8             0.880         0.25             2.5      0.104   \n",
       "672             9.8             1.240         0.34             2.0      0.079   \n",
       "684             9.8             0.980         0.32             2.3      0.078   \n",
       "1079            7.9             0.300         0.68             8.3      0.050   \n",
       "1081            7.9             0.300         0.68             8.3      0.050   \n",
       "1244            5.9             0.290         0.25            13.4      0.067   \n",
       "1493            7.7             0.540         0.26             1.9      0.089   \n",
       "1496            7.7             0.540         0.26             1.9      0.089   \n",
       "\n",
       "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "15                   51.0                 148.0  0.99860  3.17       0.93   \n",
       "109                  37.0                 153.0  0.99690  3.21       0.69   \n",
       "354                  40.5                 165.0  0.99120  3.25       0.59   \n",
       "515                  34.0                 151.0  1.00100  3.31       1.14   \n",
       "591                  23.0                 149.0  0.99220  3.12       0.50   \n",
       "636                  30.0                 147.0  0.99790  3.24       0.53   \n",
       "649                  24.0                 148.0  0.99480  3.16       0.57   \n",
       "651                  35.0                 155.0  1.00100  3.41       0.67   \n",
       "672                  32.0                 151.0  0.99800  3.15       0.53   \n",
       "684                  35.0                 152.0  0.99800  3.25       0.48   \n",
       "1079                 37.5                 278.0  0.99316  3.01       0.51   \n",
       "1081                 37.5                 289.0  0.99316  3.01       0.51   \n",
       "1244                 72.0                 160.0  0.99721  3.33       0.54   \n",
       "1493                 23.0                 147.0  0.99636  3.26       0.59   \n",
       "1496                 23.0                 147.0  0.99636  3.26       0.59   \n",
       "\n",
       "      alcohol  quality  \n",
       "15        9.2        5  \n",
       "109       9.3        5  \n",
       "354      11.9        6  \n",
       "515       9.3        5  \n",
       "591      11.5        6  \n",
       "636       9.4        5  \n",
       "649      11.3        6  \n",
       "651      11.2        5  \n",
       "672       9.5        5  \n",
       "684       9.4        5  \n",
       "1079     12.3        7  \n",
       "1081     12.3        7  \n",
       "1244     10.3        6  \n",
       "1493      9.7        5  \n",
       "1496      9.7        5  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[(data['total sulfur dioxide'] > UpperLimit)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0b3f12f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['total sulfur dioxide'][(data['total sulfur dioxide'] > 1.5*UpperLimit)] = 1.5*UpperLimit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "057fd16d",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>8.9</td>\n",
       "      <td>0.620</td>\n",
       "      <td>0.19</td>\n",
       "      <td>3.9</td>\n",
       "      <td>0.170</td>\n",
       "      <td>51.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>0.99860</td>\n",
       "      <td>3.17</td>\n",
       "      <td>0.93</td>\n",
       "      <td>9.2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>8.1</td>\n",
       "      <td>0.785</td>\n",
       "      <td>0.52</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.122</td>\n",
       "      <td>37.0</td>\n",
       "      <td>153.0</td>\n",
       "      <td>0.99690</td>\n",
       "      <td>3.21</td>\n",
       "      <td>0.69</td>\n",
       "      <td>9.3</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>354</th>\n",
       "      <td>6.1</td>\n",
       "      <td>0.210</td>\n",
       "      <td>0.40</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.066</td>\n",
       "      <td>40.5</td>\n",
       "      <td>165.0</td>\n",
       "      <td>0.99120</td>\n",
       "      <td>3.25</td>\n",
       "      <td>0.59</td>\n",
       "      <td>11.9</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>515</th>\n",
       "      <td>8.5</td>\n",
       "      <td>0.655</td>\n",
       "      <td>0.49</td>\n",
       "      <td>6.1</td>\n",
       "      <td>0.122</td>\n",
       "      <td>34.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>1.00100</td>\n",
       "      <td>3.31</td>\n",
       "      <td>1.14</td>\n",
       "      <td>9.3</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>591</th>\n",
       "      <td>6.6</td>\n",
       "      <td>0.390</td>\n",
       "      <td>0.49</td>\n",
       "      <td>1.7</td>\n",
       "      <td>0.070</td>\n",
       "      <td>23.0</td>\n",
       "      <td>149.0</td>\n",
       "      <td>0.99220</td>\n",
       "      <td>3.12</td>\n",
       "      <td>0.50</td>\n",
       "      <td>11.5</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>636</th>\n",
       "      <td>9.6</td>\n",
       "      <td>0.880</td>\n",
       "      <td>0.28</td>\n",
       "      <td>2.4</td>\n",
       "      <td>0.086</td>\n",
       "      <td>30.0</td>\n",
       "      <td>147.0</td>\n",
       "      <td>0.99790</td>\n",
       "      <td>3.24</td>\n",
       "      <td>0.53</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>649</th>\n",
       "      <td>6.7</td>\n",
       "      <td>0.420</td>\n",
       "      <td>0.27</td>\n",
       "      <td>8.6</td>\n",
       "      <td>0.068</td>\n",
       "      <td>24.0</td>\n",
       "      <td>148.0</td>\n",
       "      <td>0.99480</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.57</td>\n",
       "      <td>11.3</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>651</th>\n",
       "      <td>9.8</td>\n",
       "      <td>0.880</td>\n",
       "      <td>0.25</td>\n",
       "      <td>2.5</td>\n",
       "      <td>0.104</td>\n",
       "      <td>35.0</td>\n",
       "      <td>155.0</td>\n",
       "      <td>1.00100</td>\n",
       "      <td>3.41</td>\n",
       "      <td>0.67</td>\n",
       "      <td>11.2</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>672</th>\n",
       "      <td>9.8</td>\n",
       "      <td>1.240</td>\n",
       "      <td>0.34</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.079</td>\n",
       "      <td>32.0</td>\n",
       "      <td>151.0</td>\n",
       "      <td>0.99800</td>\n",
       "      <td>3.15</td>\n",
       "      <td>0.53</td>\n",
       "      <td>9.5</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>684</th>\n",
       "      <td>9.8</td>\n",
       "      <td>0.980</td>\n",
       "      <td>0.32</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.078</td>\n",
       "      <td>35.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>0.99800</td>\n",
       "      <td>3.25</td>\n",
       "      <td>0.48</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1079</th>\n",
       "      <td>7.9</td>\n",
       "      <td>0.300</td>\n",
       "      <td>0.68</td>\n",
       "      <td>8.3</td>\n",
       "      <td>0.050</td>\n",
       "      <td>37.5</td>\n",
       "      <td>217.5</td>\n",
       "      <td>0.99316</td>\n",
       "      <td>3.01</td>\n",
       "      <td>0.51</td>\n",
       "      <td>12.3</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1081</th>\n",
       "      <td>7.9</td>\n",
       "      <td>0.300</td>\n",
       "      <td>0.68</td>\n",
       "      <td>8.3</td>\n",
       "      <td>0.050</td>\n",
       "      <td>37.5</td>\n",
       "      <td>217.5</td>\n",
       "      <td>0.99316</td>\n",
       "      <td>3.01</td>\n",
       "      <td>0.51</td>\n",
       "      <td>12.3</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1244</th>\n",
       "      <td>5.9</td>\n",
       "      <td>0.290</td>\n",
       "      <td>0.25</td>\n",
       "      <td>13.4</td>\n",
       "      <td>0.067</td>\n",
       "      <td>72.0</td>\n",
       "      <td>160.0</td>\n",
       "      <td>0.99721</td>\n",
       "      <td>3.33</td>\n",
       "      <td>0.54</td>\n",
       "      <td>10.3</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1493</th>\n",
       "      <td>7.7</td>\n",
       "      <td>0.540</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.089</td>\n",
       "      <td>23.0</td>\n",
       "      <td>147.0</td>\n",
       "      <td>0.99636</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.59</td>\n",
       "      <td>9.7</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1496</th>\n",
       "      <td>7.7</td>\n",
       "      <td>0.540</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.089</td>\n",
       "      <td>23.0</td>\n",
       "      <td>147.0</td>\n",
       "      <td>0.99636</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.59</td>\n",
       "      <td>9.7</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "15              8.9             0.620         0.19             3.9      0.170   \n",
       "109             8.1             0.785         0.52             2.0      0.122   \n",
       "354             6.1             0.210         0.40             1.4      0.066   \n",
       "515             8.5             0.655         0.49             6.1      0.122   \n",
       "591             6.6             0.390         0.49             1.7      0.070   \n",
       "636             9.6             0.880         0.28             2.4      0.086   \n",
       "649             6.7             0.420         0.27             8.6      0.068   \n",
       "651             9.8             0.880         0.25             2.5      0.104   \n",
       "672             9.8             1.240         0.34             2.0      0.079   \n",
       "684             9.8             0.980         0.32             2.3      0.078   \n",
       "1079            7.9             0.300         0.68             8.3      0.050   \n",
       "1081            7.9             0.300         0.68             8.3      0.050   \n",
       "1244            5.9             0.290         0.25            13.4      0.067   \n",
       "1493            7.7             0.540         0.26             1.9      0.089   \n",
       "1496            7.7             0.540         0.26             1.9      0.089   \n",
       "\n",
       "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "15                   51.0                 148.0  0.99860  3.17       0.93   \n",
       "109                  37.0                 153.0  0.99690  3.21       0.69   \n",
       "354                  40.5                 165.0  0.99120  3.25       0.59   \n",
       "515                  34.0                 151.0  1.00100  3.31       1.14   \n",
       "591                  23.0                 149.0  0.99220  3.12       0.50   \n",
       "636                  30.0                 147.0  0.99790  3.24       0.53   \n",
       "649                  24.0                 148.0  0.99480  3.16       0.57   \n",
       "651                  35.0                 155.0  1.00100  3.41       0.67   \n",
       "672                  32.0                 151.0  0.99800  3.15       0.53   \n",
       "684                  35.0                 152.0  0.99800  3.25       0.48   \n",
       "1079                 37.5                 217.5  0.99316  3.01       0.51   \n",
       "1081                 37.5                 217.5  0.99316  3.01       0.51   \n",
       "1244                 72.0                 160.0  0.99721  3.33       0.54   \n",
       "1493                 23.0                 147.0  0.99636  3.26       0.59   \n",
       "1496                 23.0                 147.0  0.99636  3.26       0.59   \n",
       "\n",
       "      alcohol  quality  \n",
       "15        9.2        5  \n",
       "109       9.3        5  \n",
       "354      11.9        6  \n",
       "515       9.3        5  \n",
       "591      11.5        6  \n",
       "636       9.4        5  \n",
       "649      11.3        6  \n",
       "651      11.2        5  \n",
       "672       9.5        5  \n",
       "684       9.4        5  \n",
       "1079     12.3        7  \n",
       "1081     12.3        7  \n",
       "1244     10.3        6  \n",
       "1493      9.7        5  \n",
       "1496      9.7        5  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[(data['total sulfur dioxide'] > UpperLimit)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4ff2adb5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "fixed acidity           1599\n",
       "volatile acidity        1599\n",
       "citric acid             1599\n",
       "residual sugar          1599\n",
       "chlorides               1599\n",
       "free sulfur dioxide     1599\n",
       "total sulfur dioxide    1599\n",
       "density                 1599\n",
       "pH                      1599\n",
       "sulphates               1599\n",
       "alcohol                 1599\n",
       "quality                 1599\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "1f988dc4",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>fixed acidity</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.256131</td>\n",
       "      <td>0.671703</td>\n",
       "      <td>0.114777</td>\n",
       "      <td>0.093705</td>\n",
       "      <td>-0.153794</td>\n",
       "      <td>-0.114374</td>\n",
       "      <td>0.668047</td>\n",
       "      <td>-0.682978</td>\n",
       "      <td>0.183006</td>\n",
       "      <td>-0.061668</td>\n",
       "      <td>0.124052</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>volatile acidity</th>\n",
       "      <td>-0.256131</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.552496</td>\n",
       "      <td>0.001918</td>\n",
       "      <td>0.061298</td>\n",
       "      <td>-0.010504</td>\n",
       "      <td>0.080937</td>\n",
       "      <td>0.022026</td>\n",
       "      <td>0.234937</td>\n",
       "      <td>-0.260987</td>\n",
       "      <td>-0.202288</td>\n",
       "      <td>-0.390558</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>citric acid</th>\n",
       "      <td>0.671703</td>\n",
       "      <td>-0.552496</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.143577</td>\n",
       "      <td>0.203823</td>\n",
       "      <td>-0.060978</td>\n",
       "      <td>0.030744</td>\n",
       "      <td>0.364947</td>\n",
       "      <td>-0.541904</td>\n",
       "      <td>0.312770</td>\n",
       "      <td>0.109903</td>\n",
       "      <td>0.226373</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>residual sugar</th>\n",
       "      <td>0.114777</td>\n",
       "      <td>0.001918</td>\n",
       "      <td>0.143577</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.055610</td>\n",
       "      <td>0.187049</td>\n",
       "      <td>0.195846</td>\n",
       "      <td>0.355283</td>\n",
       "      <td>-0.085652</td>\n",
       "      <td>0.005527</td>\n",
       "      <td>0.042075</td>\n",
       "      <td>0.013732</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>chlorides</th>\n",
       "      <td>0.093705</td>\n",
       "      <td>0.061298</td>\n",
       "      <td>0.203823</td>\n",
       "      <td>0.055610</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.005562</td>\n",
       "      <td>0.050188</td>\n",
       "      <td>0.200632</td>\n",
       "      <td>-0.265026</td>\n",
       "      <td>0.371260</td>\n",
       "      <td>-0.221141</td>\n",
       "      <td>-0.128907</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <td>-0.153794</td>\n",
       "      <td>-0.010504</td>\n",
       "      <td>-0.060978</td>\n",
       "      <td>0.187049</td>\n",
       "      <td>0.005562</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.673055</td>\n",
       "      <td>-0.021946</td>\n",
       "      <td>0.070377</td>\n",
       "      <td>0.051658</td>\n",
       "      <td>-0.069408</td>\n",
       "      <td>-0.050656</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <td>-0.114374</td>\n",
       "      <td>0.080937</td>\n",
       "      <td>0.030744</td>\n",
       "      <td>0.195846</td>\n",
       "      <td>0.050188</td>\n",
       "      <td>0.673055</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.077256</td>\n",
       "      <td>-0.062581</td>\n",
       "      <td>0.045862</td>\n",
       "      <td>-0.213432</td>\n",
       "      <td>-0.192365</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>density</th>\n",
       "      <td>0.668047</td>\n",
       "      <td>0.022026</td>\n",
       "      <td>0.364947</td>\n",
       "      <td>0.355283</td>\n",
       "      <td>0.200632</td>\n",
       "      <td>-0.021946</td>\n",
       "      <td>0.077256</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.341699</td>\n",
       "      <td>0.148506</td>\n",
       "      <td>-0.496180</td>\n",
       "      <td>-0.174919</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pH</th>\n",
       "      <td>-0.682978</td>\n",
       "      <td>0.234937</td>\n",
       "      <td>-0.541904</td>\n",
       "      <td>-0.085652</td>\n",
       "      <td>-0.265026</td>\n",
       "      <td>0.070377</td>\n",
       "      <td>-0.062581</td>\n",
       "      <td>-0.341699</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.196648</td>\n",
       "      <td>0.205633</td>\n",
       "      <td>-0.057731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sulphates</th>\n",
       "      <td>0.183006</td>\n",
       "      <td>-0.260987</td>\n",
       "      <td>0.312770</td>\n",
       "      <td>0.005527</td>\n",
       "      <td>0.371260</td>\n",
       "      <td>0.051658</td>\n",
       "      <td>0.045862</td>\n",
       "      <td>0.148506</td>\n",
       "      <td>-0.196648</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.093595</td>\n",
       "      <td>0.251397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>alcohol</th>\n",
       "      <td>-0.061668</td>\n",
       "      <td>-0.202288</td>\n",
       "      <td>0.109903</td>\n",
       "      <td>0.042075</td>\n",
       "      <td>-0.221141</td>\n",
       "      <td>-0.069408</td>\n",
       "      <td>-0.213432</td>\n",
       "      <td>-0.496180</td>\n",
       "      <td>0.205633</td>\n",
       "      <td>0.093595</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.476166</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>quality</th>\n",
       "      <td>0.124052</td>\n",
       "      <td>-0.390558</td>\n",
       "      <td>0.226373</td>\n",
       "      <td>0.013732</td>\n",
       "      <td>-0.128907</td>\n",
       "      <td>-0.050656</td>\n",
       "      <td>-0.192365</td>\n",
       "      <td>-0.174919</td>\n",
       "      <td>-0.057731</td>\n",
       "      <td>0.251397</td>\n",
       "      <td>0.476166</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      fixed acidity  volatile acidity  citric acid  \\\n",
       "fixed acidity              1.000000         -0.256131     0.671703   \n",
       "volatile acidity          -0.256131          1.000000    -0.552496   \n",
       "citric acid                0.671703         -0.552496     1.000000   \n",
       "residual sugar             0.114777          0.001918     0.143577   \n",
       "chlorides                  0.093705          0.061298     0.203823   \n",
       "free sulfur dioxide       -0.153794         -0.010504    -0.060978   \n",
       "total sulfur dioxide      -0.114374          0.080937     0.030744   \n",
       "density                    0.668047          0.022026     0.364947   \n",
       "pH                        -0.682978          0.234937    -0.541904   \n",
       "sulphates                  0.183006         -0.260987     0.312770   \n",
       "alcohol                   -0.061668         -0.202288     0.109903   \n",
       "quality                    0.124052         -0.390558     0.226373   \n",
       "\n",
       "                      residual sugar  chlorides  free sulfur dioxide  \\\n",
       "fixed acidity               0.114777   0.093705            -0.153794   \n",
       "volatile acidity            0.001918   0.061298            -0.010504   \n",
       "citric acid                 0.143577   0.203823            -0.060978   \n",
       "residual sugar              1.000000   0.055610             0.187049   \n",
       "chlorides                   0.055610   1.000000             0.005562   \n",
       "free sulfur dioxide         0.187049   0.005562             1.000000   \n",
       "total sulfur dioxide        0.195846   0.050188             0.673055   \n",
       "density                     0.355283   0.200632            -0.021946   \n",
       "pH                         -0.085652  -0.265026             0.070377   \n",
       "sulphates                   0.005527   0.371260             0.051658   \n",
       "alcohol                     0.042075  -0.221141            -0.069408   \n",
       "quality                     0.013732  -0.128907            -0.050656   \n",
       "\n",
       "                      total sulfur dioxide   density        pH  sulphates  \\\n",
       "fixed acidity                    -0.114374  0.668047 -0.682978   0.183006   \n",
       "volatile acidity                  0.080937  0.022026  0.234937  -0.260987   \n",
       "citric acid                       0.030744  0.364947 -0.541904   0.312770   \n",
       "residual sugar                    0.195846  0.355283 -0.085652   0.005527   \n",
       "chlorides                         0.050188  0.200632 -0.265026   0.371260   \n",
       "free sulfur dioxide               0.673055 -0.021946  0.070377   0.051658   \n",
       "total sulfur dioxide              1.000000  0.077256 -0.062581   0.045862   \n",
       "density                           0.077256  1.000000 -0.341699   0.148506   \n",
       "pH                               -0.062581 -0.341699  1.000000  -0.196648   \n",
       "sulphates                         0.045862  0.148506 -0.196648   1.000000   \n",
       "alcohol                          -0.213432 -0.496180  0.205633   0.093595   \n",
       "quality                          -0.192365 -0.174919 -0.057731   0.251397   \n",
       "\n",
       "                       alcohol   quality  \n",
       "fixed acidity        -0.061668  0.124052  \n",
       "volatile acidity     -0.202288 -0.390558  \n",
       "citric acid           0.109903  0.226373  \n",
       "residual sugar        0.042075  0.013732  \n",
       "chlorides            -0.221141 -0.128907  \n",
       "free sulfur dioxide  -0.069408 -0.050656  \n",
       "total sulfur dioxide -0.213432 -0.192365  \n",
       "density              -0.496180 -0.174919  \n",
       "pH                    0.205633 -0.057731  \n",
       "sulphates             0.093595  0.251397  \n",
       "alcohol               1.000000  0.476166  \n",
       "quality               0.476166  1.000000  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "291b8141",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='count'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.countplot(x = 'quality', data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "cbf549d8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='fixed acidity'>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"fixed acidity\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "29c5d2c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='volatile acidity'>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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EELEIUn+QggRhyT5CyFEHJLkL4QQcUiSvxBCxCFJ/kIIEYck+QshRBxSIpHev0ZqaWkpRqOxp8MQQog+xefzMWbMmNNu6xPJXwghRNeSZh8hhIhDkvyFECIOSfIXQog4JMlfCCHikCR/IYSIQ5L8hRAiDvXbYi6hUIif//znHDx4EE3TWLx4MXl5eT0dVrdoaGhg5syZPPvssxQVFfV0ODE3Y8YMEhMTAcjNzWXx4sU9HFHs/f73v2fNmjUEAgHmzZvH7NmzezqkmFqxYgWvv/46EB27XlZWxvr160lKSurhyGIjEAhw3333cfToUVRV5dFHH+3yf8v9Nvm///77ACxbtowNGzawePFinnnmmR6OKvYCgQAPPfQQJpOpp0PpFj6fD4Dnn3++hyPpPhs2bGDLli28/PLLeDwenn322Z4OKeZmzpzJzJkzAXjkkUeYNWtWv038AB9++CHBYJBly5axfv16fvWrX7F06dIuPUa/bfb56le/yqOPPgpAZWUlaWlpPRxR93jiiSeYO3cuGRkZPR1Kt9i9ezcej4dvf/vb3HrrrZSWlvZ0SDH30UcfUVxczI9+9CO+//3vM3ny5J4Oqdts376dzz//nDlz5vR0KDFVUFBAKBQiHA7jdDrR6br+Pr3f3vkD6HQ67r33Xt59912efvrpng4n5lasWEFKSgqTJk3iD3/4Q0+H0y1MJhO33XYbs2fP5tChQ3z3u9/lnXfeick/lt6iqamJyspKfve731FRUcEPfvAD3nnnHRRF6enQYu73v/89P/rRj3o6jJizWCwcPXqU66+/nqamJn73u991+TH67Z3/MU888QR///vfefDBB3G73T0dTky99tprfPzxxyxYsICysjLuvfde6urqejqsmCooKOBrX/saiqJQUFCA3W7v9+dst9u54oorMBgMFBYWYjQaaWxs7OmwYs7hcHDgwAEmTpzY06HE3F/+8heuuOIK/v73v/O3v/2N++67r72Js6v02+T/xhtv8Pvf/x4As9mMoihomtbDUcXWiy++yAsvvMDzzz9PSUkJTzzxBOnp6T0dVky9+uqrLFmyBICamhqcTme/P+fx48ezbt06IpEINTU1eDwe7HZ7T4cVcxs3buTyyy/v6TC6RVJSUvsgBpvNRjAYJBQKdekx+u2z8TXXXMP999/P17/+dYLBIA888ICsDNoP3Xzzzdx///3MmzcPRVFYtGhRv27yAbjqqqvYuHEjN998M5FIhIceeqjf39gAHDx4kNzc3J4Oo1v827/9Gw888ADz588nEAjw05/+FIvF0qXHkFU9hRAiDvXbZh8hhBBnJslfCCHikCR/IYSIQ5L8hRAiDknyF0KIOCTJX4gusHTpUl5++WXKysr4zW9+A8C7775LTU1ND0cmxOlJ8heiC5WUlHD77bcD8Nxzz+F0Ons4IiFOr3/PhhHiHLlcLu666y4cDgeDBw9my5Yt2O12Hn74YYqKinj55Zepr6/nxz/+MU899RQ7duzA5XJRVFR00hLSGzZsYNmyZdx4443tS2wcW3fo3nvvJRQKMWPGDF577TUMBkMPnrGId3LnLwTw0ksvMXToUF566SVmzJiBy+U67fucTidJSUn8+c9/ZtmyZZSWlp62aWfy5MntS2xMmzaNf/7zn4RCIdatW8ell14qiV/0OLnzFwKoqKhg0qRJAIwbN+6U5HxsIvyxRdTuvPNOLBYLbrebQCBw1n1brVYuvvhiPvroI1asWMEPf/jD2JyEEOdB7vyFAIYOHcpnn30GwJ49e/D7/RgMhvYVQnft2gXA2rVrqaqq4r//+7+588478Xq9nGmFFEVR2rfdcsstvPLKKzQ0NHDRRRd1wxkJcXaS/IUAZs+eTX19PV//+tf53//9XwBuvfVWFi5cyG233da+ouKoUaM4cuQIt9xyC3fccQcDBw6ktrb2tPscO3Ys99xzD83NzYwePZry8nKmT5/ebeckxNnIwm5CfIHP5+P6669nzZo1XbbPcDjMvHnz+NOf/oTVau2y/QrRWXLnL0SMHTlyhJtuuokbb7xREr/oNeTOXwgh4pDc+QshRByS5C+EEHFIkr8QQsQhSf5CCBGHJPkLIUQc+v8CtYIyhKQ18gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"volatile acidity\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "09c69c14",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='volatile acidity'>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"volatile acidity\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "5c6dbda4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='citric acid'>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"citric acid\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "8c9ff4a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='residual sugar'>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"residual sugar\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "959505bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='chlorides'>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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rARCKpVi06ZNVFdXs2HDBhobG4c9vmvXLq6//nqqq6v55S9/mbn/4Ycfprq6muuuu47HH388G00TQggxgqwMGe3cuZNYLMb27dupqalh69atPPjggwDE43G2bNnCr371KxwOBzfccAOXX345R44c4c033+TRRx8lHA7zs5/9LBtNE0IIMYKsBMLevXtZs2YNACtWrKC2tjbzWH19PfPnz8fn8wGwatUqXn/9dQ4cOMDixYu55ZZbCAQC3HrrrdlomhBCiBFkJRACgQButztzW9M0EokEuq4TCATweDyZx1wuF4FAgJ6eHlpbW3nooYdobm7m85//PM8+++yohaCi0Sh1dXXjamMkEhn3sflKzrkwyDlPf9k636wEgtvtJhgMZm6nUil0XT/tY8FgEI/Hg9/vZ9GiRVitVhYtWoTNZqO7u5uSkpIR38dms1FVVTWuNtbV1Y372Hwl51wY5Jynv3M935HCJCuTyitXrmT37t0A1NTUsHjx4sxjlZWVNDY20tvbSywW4/XXX+eSSy5h1apVvPTSSxiGQVtbG+FwGL/fn43mCSGEOI2s9BDWrl3Lnj17WL9+PYZhsHnzZnbs2EEoFKK6uprbb7+dm2++GcMwuP7665kxYwYzZszgtddeY926dRiGwaZNm9C0/LmwhBBC5DvFMAwj140Yr5qaGmw2W66bIYQQeSUajbJixYpT7s/rQBBCCDFxZKeyEEIIQAJBCCFEmgSCEEIIQAJBCCFEmgSCEEIIQAJBCCFEWsFdICeZTPK1r32NhoYGNE1jy5YtzJ8/P9fNyrquri6uu+46fvazn1FZWZnr5mTdNddck6mZNXfuXLZs2ZLjFmXfww8/zK5du4jH49xwww184hOfyHWTsurJJ5/k17/+NXCirtmePXvwer05bln2xONxbr/9dlpaWlBVlbvvvntCf58LLhBefPFFAB577DFeffVVtmzZkinNPV3F43E2bdqE3W7PdVMmRTQaBWDbtm05bsnkefXVVwuufPx1113HddddB8A3v/lNrr/++mkdBgB/+MMfSCQSPPbYY+zZs4f77ruP+++/f8Jev+CGjD784Q9z9913A9Da2kppaWmOW5R999xzD+vXr6e8vDzXTZkU77zzDuFwmM9+9rN86lOfoqamJtdNyrqXX345Uz7+c5/7HJdddlmumzRp9u/fz7vvvkt1dXWum5J1FRUVJJNJUqkUgUAgUzR0ohRcDwFA13Vuu+02nn/+eX7wgx/kujlZ9eSTT1JcXMyaNWv48Y9/nOvmTAq73c7NN9/MJz7xCY4ePcrf/d3f8eyzz074L89UMp7y8dPFww8/zC233JLrZkwKp9NJS0sLV1xxBT09PTz00EMT+voF10MYdM899/Dcc8/x9a9/nVAolOvmZM0TTzzBH//4RzZs2EBdXR233XYbHR0duW5WVlVUVHD11VejKAoVFRX4/f5pf85+v5/Vq1efUj5+uuvv7+fIkSO8//3vz3VTJsXPf/5zVq9ezXPPPcdvfvMbbr/99swQ6UQouED4n//5Hx5++GEAHA4HiqJM66qqv/jFL/jv//5vtm3bRlVVFffccw9lZWW5blZW/epXv2Lr1q0AtLW1EQgEpv05F2r5+Ndee42/+qu/ynUzJo3X680slvD5fCQSCZLJ5IS9/vTtQ4/gIx/5CF/5yle46aabSCQS3HHHHVIxdZpZt24dX/nKV7jhhhtQFIXNmzdP6+EigMsvv7wgy8c3NDQwd+7cXDdj0nzmM5/hjjvu4MYbbyQej/PFL34Rp9M5Ya8v1U6FEEIABThkJIQQ4vQkEIQQQgASCEIIIdIkEIQQQgASCEIIIdIkEITIovvvv59HH32Uuro6fvjDHwLw/PPP09bWluOWCXEqCQQhJkFVVRX/+I//CMB//dd/EQgEctwiIU41vXfrCHGOgsEgGzdupL+/n/POO48333wTv9/PnXfeSWVlJY8++iidnZ380z/9E9/73veora0lGAxSWVk5rOT2q6++ymOPPcbHP/7xTAmRwVpLt912G8lkkmuuuYYnnngCq9WawzMWhUx6CEKM4pFHHmHJkiU88sgjXHPNNQSDwdM+LxAI4PV6+Y//+A8ee+wxampqTjssdNlll2VKiHzsYx/jhRdeIJlM8tJLL3HppZdKGIickh6CEKNobm5mzZo1AKxcufKUD+zBjf6DxeS+9KUv4XQ6CYVCxOPxUV/b7Xbzvve9j5dffpknn3ySf/iHf8jOSQgxRtJDEGIUS5Ys4Y033gDg4MGDxGIxrFZrpnrqgQMHANi9ezfHjh3j+9//Pl/60peIRCKMVBVGUZTMY5/85Cd5/PHH6erqYunSpZNwRkKMTAJBiFF84hOfoLOzk5tuuol///d/B+BTn/oUd911FzfffHOm0uTy5ctpamrik5/8JF/4wheYN28e7e3tp33NSy65hFtvvZXe3l4uvvhiGhsbueqqqybtnIQYiRS3E2KMotEoV1xxBbt27Zqw10ylUtxwww389Kc/xe12T9jrCjEe0kMQIkeampq49tpr+fjHPy5hIKYE6SEIIYQApIcghBAiTQJBCCEEIIEghBAiTQJBCCEEIIEghBAi7f8HLHW36vRyQHUAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"chlorides\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7ecf9669",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='free sulfur dioxide'>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"free sulfur dioxide\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f161ac78",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='total sulfur dioxide'>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"total sulfur dioxide\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "b3894eb9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='density'>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"density\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "01ea8443",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='pH'>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"pH\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "19c0283d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='sulphates'>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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fvWraJrGN7/5TVkxdYy58847eeyxx1i+fDkWi4W1a9eO6eEigM9+9rO8//773HnnnZimyRNPPDGmP+j0amxspKSkZLibMWT+6q/+im9+85vcfffdqKrKww8/TGZm5qC9vqx2KoQQAhiHQ0ZCCCEuTAJBCCEEIIEghBCihwSCEEIIQAJBCCFEDwkEIdLoBz/4AT//+c+pr6/nhz/8IQC//e1vaW5uHuaWCXE+CQQhhkBlZSUPPfQQAD/96U8JhULD3CIhzje279YR4jKFw2EeeeQRgsEg06dPZ/fu3fj9flatWsW0adP4+c9/TltbG3/3d3/HM888Q11dHeFwmGnTpp215PbOnTt55ZVXuPXWW5NLiPSutfToo4+i6zqLFy/mF7/4BU6ncxiPWIxn0kMQog+bNm3iiiuuYNOmTSxevJhwOHzB54VCIbKysnjxxRd55ZVXqK2tveCw0IIFC5JLiCxatIi33noLXdd5++23uf766yUMxLCSHoIQfTh+/Djz588HYM6cOeedsHtv9O9dTG7lypVkZmYSiURQVbXP1/Z6vcydO5d33nmH1157jb/9279Nz0EIkSLpIQjRhyuuuII//elPAOzfvx9FUXA6ncnVU/fu3QvAjh07OHXqFN/73vdYuXIlsViMi60KY7FYktvuuusuXn31Vdrb25k5c+YQHJEQFyeBIEQflixZQltbG1/96lf593//dwBWrFjB6tWrue+++5IrTV5zzTUcO3aMu+66i7//+7+ntLSUlpaWC77m7Nmz+ad/+icCgQCzZs3iyJEjfPnLXx6yYxLiYmRxOyFSFI/Huemmm9i2bdugvaZhGCxfvpyf/OQneL3eQXtdIQZCeghCDJNjx45x2223ceutt0oYiBFBeghCCCEA6SEIIYToIYEghBACkEAQQgjRQwJBCCEEIIEghBCix/8PlQ7SFKB6lWIAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"sulphates\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "de40eb99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='alcohol'>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.set_style('whitegrid')\n",
    "sns.lineplot(data=data, x=\"quality\", y=\"alcohol\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f7c69b35",
   "metadata": {},
   "outputs": [],
   "source": [
    "bins = (2, 6.5, 8)\n",
    "group_names = ['bad', 'good']\n",
    "data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "15e3a609",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='quality', ylabel='count'>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
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
    "sns.countplot(x = 'quality', data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "7614c08b",
   "metadata": {},
   "outputs": [],
   "source": [
    "labelencoder = LabelEncoder()\n",
    "data['quality'] = labelencoder.fit_transform(data['quality'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "17eddb48",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.88</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.098</td>\n",
       "      <td>25.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>0.9968</td>\n",
       "      <td>3.20</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.8</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.76</td>\n",
       "      <td>0.04</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.092</td>\n",
       "      <td>15.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0.9970</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.65</td>\n",
       "      <td>9.8</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.2</td>\n",
       "      <td>0.28</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.075</td>\n",
       "      <td>17.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>0.9980</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.58</td>\n",
       "      <td>9.8</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "0            7.4              0.70         0.00             1.9      0.076   \n",
       "1            7.8              0.88         0.00             2.6      0.098   \n",
       "2            7.8              0.76         0.04             2.3      0.092   \n",
       "3           11.2              0.28         0.56             1.9      0.075   \n",
       "4            7.4              0.70         0.00             1.9      0.076   \n",
       "\n",
       "   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "0                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "1                 25.0                  67.0   0.9968  3.20       0.68   \n",
       "2                 15.0                  54.0   0.9970  3.26       0.65   \n",
       "3                 17.0                  60.0   0.9980  3.16       0.58   \n",
       "4                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "\n",
       "   alcohol  quality  \n",
       "0      9.4        0  \n",
       "1      9.8        0  \n",
       "2      9.8        0  \n",
       "3      9.8        0  \n",
       "4      9.4        0  "
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "84847fea",
   "metadata": {},
   "outputs": [],
   "source": [
    "target = 'quality'\n",
    "x = data.drop(['quality'], axis=1)\n",
    "y = data[target]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "bb078106",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "c82a0288",
   "metadata": {},
   "outputs": [],
   "source": [
    "qqx_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "09253d9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "049cf232",
   "metadata": {},
   "outputs": [],
   "source": [
    "LogisticRegression = LogisticRegression(C=0.7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "34de8afe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=0.7)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LogisticRegression.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "1b860977",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Acuracy of logistic regression for test set:  86.25\n",
      "Classification Report : \n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.97      0.92       273\n",
      "           1       0.59      0.21      0.31        47\n",
      "\n",
      "    accuracy                           0.86       320\n",
      "   macro avg       0.73      0.59      0.62       320\n",
      "weighted avg       0.84      0.86      0.83       320\n",
      "\n",
      "Confusion Matrix : \n",
      " [[266   7]\n",
      " [ 37  10]]\n",
      "\n",
      "Confusion Matrix heatmap : \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"Acuracy of logistic regression for test set: \", accuracy_score(y_test, LogisticRegression.predict(x_test))*100)\n",
    "print('Classification Report : \\n', classification_report(y_test, LogisticRegression.predict(x_test)))\n",
    "print(\"Confusion Matrix : \\n\", confusion_matrix(y_test, LogisticRegression.predict(x_test)))\n",
    "print('\\nConfusion Matrix heatmap : ')\n",
    "sns.heatmap(confusion_matrix(y_test, LogisticRegression.predict(x_test)), annot=True, fmt='d')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "c143f667",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(n_estimators=75)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rfc = RandomForestClassifier(n_estimators=75)\n",
    "rfc.fit(x_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "ab020ff3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Acuracy of random forest for test set:  88.125\n",
      "\n",
      "Classification Report for test case : \n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.92      0.95      0.93       273\n",
      "           1       0.62      0.49      0.55        47\n",
      "\n",
      "    accuracy                           0.88       320\n",
      "   macro avg       0.77      0.72      0.74       320\n",
      "weighted avg       0.87      0.88      0.88       320\n",
      "\n",
      "\n",
      "Confusion Matrix for test case : \n",
      " [[259  14]\n",
      " [ 24  23]]\n",
      "\n",
      "Confusion Matrix Heatamp for test case: \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"\\nAcuracy of random forest for test set: \", accuracy_score(y_test, rfc.predict(x_test))*100)\n",
    "print('\\nClassification Report for test case : \\n', classification_report(y_test, rfc.predict(x_test)))\n",
    "print(\"\\nConfusion Matrix for test case : \\n\", confusion_matrix(y_test, rfc.predict(x_test)))\n",
    "print('\\nConfusion Matrix Heatamp for test case: ')\n",
    "sns.heatmap(confusion_matrix(y_test, rfc.predict(x_test)), annot=True, fmt='d')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "92d1a3c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "pickle.dump(rfc, open('winemodel.pkl', 'wb'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6948e218",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
