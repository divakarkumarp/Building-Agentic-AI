import openai

# Set up your OpenAI API key
openai.api_key = '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'

try:
    # Make a simple API call to test the connection
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": "Capital of india"}
        ]
    )

    # Print the response
    print("API is working correctly!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("Error:", e)
