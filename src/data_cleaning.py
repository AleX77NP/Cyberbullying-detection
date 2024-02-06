import re
import string
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
import contractions


# Clean emojis from text
def strip_emoji(text):
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", text)


# Remove punctuations, stopwords, links, mentions and new line characters
def strip_all_entities(text, stop_words):
    text = re.sub(
        r"\r|\n", " ", text.lower()
    )  # Replace newline and carriage return with space, and convert to lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # Remove links and mentions
    text = re.sub(r"[^\x00-\x7f]", "", text)  # Remove non-ASCII characters
    banned_list = string.punctuation
    table = str.maketrans("", "", banned_list)
    text = text.translate(table)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


# Clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    # Remove hashtags at the end of the sentence
    new_tweet = re.sub(r"(\s+#[\w-]+)+\s*$", "", tweet).strip()

    # Remove the # symbol from hashtags in the middle of the sentence
    new_tweet = re.sub(r"#([\w-]+)", r"\1", new_tweet).strip()

    return new_tweet


# Filter special characters such as & and $ present in some words
def filter_chars(text):
    return " ".join(
        "" if ("$" in word) or ("&" in word) else word for word in text.split()
    )


# Remove multiple spaces
def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)


# Function to check if the text is in English, and return an empty string if it's not
def filter_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""


# Expand contractions
def expand_contractions(text):
    return contractions.fix(text)


# Remove numbers
def remove_numbers(text):
    return re.sub(r"\d+", "", text)


# Lemmatize words
def lemmatize(text, lemmatizer):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


# Remove short words
def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return " ".join(long_words)


# Replace elongated words with their base form
def replace_elongated_words(text):
    regex_pattern = r"\b(\w+)((\w)\3{2,})(\w*)\b"
    return re.sub(regex_pattern, r"\1\3\4", text)


# Remove repeated punctuation
def remove_repeated_punctuation(text):
    return re.sub(r"[\?\.\!]+(?=[\?\.\!])", "", text)


# Remove extra whitespace
def remove_extra_whitespace(text):
    return " ".join(text.split())


def remove_url_shorteners(text):
    return re.sub(
        r"(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+",
        "",
        text,
    )


# Remove spaces at the beginning and end of the tweet
def remove_spaces_from_tweet(tweet):
    return tweet.strip()


# Remove short tweets
def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""


# Call all cleaning methods inside this one
def clean_tweet(tweet, lemmatizer, stop_words):
    # tweet = strip_emoji(tweet)
    tweet = expand_contractions(tweet)
    tweet = filter_non_english(tweet)
    tweet = strip_all_entities(tweet, stop_words)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize(tweet, lemmatizer)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_from_tweet(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = " ".join(tweet.split())  # Remove multiple spaces between words
    return tweet
