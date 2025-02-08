def get_gpt_sentiment(client, prompt):    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content.lower().strip()

def get_gemini_sentiment(model, prompt):
    response = model.generate_content(prompt)
    return response.text.lower().strip()

def get_roberta_sentiment(classifier, text):
    results = classifier(text)
    max_score = -1
    max_label = ''
    for pred in results[0]:
        if pred['score'] > max_score:
            max_score = pred['score']
            max_label = pred['label']
    return max_label.lower()