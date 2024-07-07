import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rake_nltk import Rake
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Load the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

courses_df = pd.read_excel("C:/Users/vinny/Documents/Humber Assignments/Semester 2/Capstone project/Rasa/actions/Course_Dataset.xlsx")

def calculate_similarity(job_description, program_details):
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    program_embedding = model.encode(program_details, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(job_embedding, program_embedding).item()
    return similarity * 100

def extract_keywords(text, num_keywords=15):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:num_keywords]
    return keywords

def find_common_keywords(job_description, program_details, num_keywords=15):
    job_keywords = extract_keywords(job_description, num_keywords)
    program_keywords = extract_keywords(program_details, num_keywords)
    common_keywords = set(job_keywords).intersection(set(program_keywords))
    return list(common_keywords)

class ActionProvideRecommendations(Action):
    def name(self) -> str:
        return "action_provide_recommendations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        job_description = tracker.get_slot('job_description') or tracker.latest_message.get('text')
        additional_info = tracker.get_slot('additional_info')
        
        if additional_info:
            job_description += " " + additional_info

        program_matches = []
        for _, row in courses_df.iterrows():
            program_details = f"{row['PROGRAM NAME']} {row['PROGRAM OVERVIEW']} {row.get('COURSES', '')} {row.get('LEARNING OUTCOME', '')} {row.get('YOUR CAREER', '')}"
            match_percentage = calculate_similarity(job_description, program_details)
            common_keywords = find_common_keywords(job_description, program_details)
            program_matches.append({
                "program_name": row['PROGRAM NAME'],
                "match_percentage": match_percentage,
                "common_keywords": common_keywords
            })

        sorted_programs = sorted(program_matches, key=lambda x: x["match_percentage"], reverse=True)
        top_programs = sorted_programs[:10]

        recommendations = ""
        for program in top_programs:
            recommendations += f"Program: {program['program_name']}\n"
            recommendations += f"Match Percentage: {program['match_percentage']}\n"
            recommendations += f"Common Keywords: {', '.join(program['common_keywords'])}\n\n"

        dispatcher.utter_message(text=f"Here are the top program recommendations:\n\n{recommendations}")
        return []
