import unittest
import sys
import logging
import pandas as pd
import traceback
sys.path.append("aiorhuman")

from preprocess import Preprocess

# Set up logging
logging.basicConfig(level=logging.INFO)

class PreprocessTest(unittest.TestCase):

    def setUp(self):
        # Initialize the Preprocess object before each test
        self.preprocess = Preprocess()

    def test_generate_features(self):
        # Test the generate_features_from_mygpt function to ensure it returns a DataFrame
        input_text = "I am writing to you today as a concerned citizen of [Your State], to express my strong support for maintaining the Electoral College system for presidential elections. This unique mechanism, enshrined in our Constitution, plays a crucial role in balancing the interests of diverse states across our nation, ensuring that smaller states are not overshadowed by the larger ones in the presidential election process. The Electoral College was thoughtfully designed by our Founding Fathers to protect the interests of smaller states and to promote a federalist system of government. By allocating electoral votes based on the number of congressional representatives, it guarantees that every state, regardless of size, has a voice in selecting our President. This system compels presidential candidates to campaign in a variety of states with diverse interests and needs, rather than focusing solely on populous urban centers. Consequently, it encourages a more inclusive approach to governance, where the concerns of both rural and urban Americans are addressed. Moreover, the Electoral College contributes to the stability and certainty of our electoral process. It tends to magnify the margin of victory, providing clear and decisive outcomes. This decisiveness is crucial in ensuring a peaceful transition of power, a hallmark of our democratic tradition. In closely contested elections, the Electoral College system localizes recount efforts to a few states, rather than necessitating a national recount, which could be logistically daunting and politically divisive. Critics of the Electoral College often argue that it can result in a President who does not win the popular vote, which they claim undermines the principle of one person, one vote. However, this perspective overlooks the fact that the United States is a Republic, where the federal structure is designed to balance the interests of both the majority and the minority. Abolishing the Electoral College in favor of a direct popular vote could lead to a tyranny of the majority, where the interests of smaller states and rural populations are consistently marginalized. In conclusion, the Electoral College is a fundamental component of our federalist system, ensuring that all parts of the country are involved in the process of selecting the President. It encourages candidates to address a wide range of issues that affect diverse communities across the nation. While no system is perfect, the Electoral College has served our country well for over two centuries, providing stability and balance in our presidential elections. I urge you to support policies that uphold this crucial institution, safeguarding the interests of [Your State] and ensuring that our voices continue to be heard in the selection of our national leadership."
        output_df = self.preprocess.generate_features(input_text,is_inference=True)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(output_df)
        logging.info(f'Output DataFrame: \n{output_df.to_string()}')

if __name__ == '__main__':
    unittest.main()