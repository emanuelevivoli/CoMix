base_prompt = """
The tasks you have to perform are two: captioning and element extraction.

First, provide a detailed description of the panel (a.k.a caption). Focus on the characters, their appearance, their actions, and the setting. Then, give a broad description of the panel.

After, you are requested to perform the "Attribute Extraction from a Comic Book Panel Caption" task. 
The task involves identifying and extracting all the distinct visual and textual attributes from the caption you identified, focusing on concrete and detectable attributes. 
These attributes include objects, characters, and other tangible elements that contribute to the scene's depiction. 
Consider the detailed comic panel caption, which outlines the setting, characters, attire, expressions, and any specific objects or textual elements present in the scene. 
As output, we expect the list of attributes, each presented as a noun or a noun phrase, that are mentioned in the description. These should capture all individual visual and textual elements that can be associated with a bounding box indicating their grounded position in the scene.

Steps to follow:
0. provide a detailed caption of the panel
1. Carefully read the panel description to fully understand the scene, characters involved, and setting.
2. Extract only tangible visual elements such as objects (e.g., desk, map), character descriptions (e.g., young man, older man), clothing (e.g., green pilot’s cap, military-style shirt), and physical features (e.g., blond hair, serious expression). Avoid including elements like artistic style or color schemes that cannot be visually grounded or are abstract.
3. List each attribute individually. For instance, if a character is described as a "young man with blond hair," this should be noted as "young man" and "blond hair" as separate attributes.
4. Do not repeat attributes; each unique attribute should only appear once in your output list.

We expect the output to be in the following format:
---
```caption
<here the detailed caption>
```

```list
<element 1>
<element 2>
...
```
---

Please, follow the instructions above.
"""

idefics2_prompt = """
You have to perform the task of "Captioning and object listing".

First, provide a detailed description of the panel (caption). Focus on the characters, their appearance, their actions, and the setting. Then, give a broad description of the panel.

After, you are requested to "Extraction Objects and Attributes" from a Comic Book Panel Caption. 
This task involves identifying and extracting all the <distinct visual and textual objects> from the caption you identified, focusing on concrete and detectable elements. 
These attributes include objects, characters, and other tangible elements that contribute to the scene's depiction.

Consider the detailed comic panel caption, which outlines the setting, characters, attire, expressions, and any specific objects or textual elements present in the scene. 
As output, we expect the list of attributes, each presented as a noun or a noun phrase, that are mentioned in the description. These should capture all individual visual and textual elements that can be associated with a bounding box indicating their grounded position in the scene.

Steps to follow:
0. provide a detailed caption of the panel
1. Carefully read the panel description to fully understand the scene, characters involved, and setting.
2. Extract only tangible visual elements such as objects (e.g., desk, map), character descriptions (e.g., young man, older man), clothing (e.g., green pilot’s cap, military-style shirt), and physical features (e.g., blond hair, serious expression). Avoid including elements like artistic style or color schemes that cannot be visually grounded or are abstract.
3. List each attribute individually. For instance, if a character is described as a "young man with blond hair," this should be noted as "young man" and "blond hair" as separate attributes.
4. Do not repeat attributes; each unique attribute should only appear once in your output list.

We expect the output to be in the following format:
---
```caption
<here the detailed caption>
```

```list
<element 1>
<element 2>
...
```
---

Please, follow the instructions above and USE THE OUTPUT FORMAT PROVIDED WITH ```caption``` AND ```list``` TAGS.
"""


minicpm26_prompt = """
Analyze the comic book panel and provide a detailed description with the following details:

- **CharacterList**: A list of characters that appear in the panel, including a visual description that would allow someone to identify them just by seeing an image. Include details such as physical appearance, clothing, expressions, and any distinctive features.

- **Elements in the Scene**: List all the elements present in the scene, including objects, props, and environmental details. Describe their attributes such as color, size, position, and any notable characteristics.

- **Actions**: Describe all the actions taking place in the panel. Specify which characters are involved in each action and provide details about their movements, gestures, and interactions with other characters or objects.

- **Important Objects**: Identify any important objects in the scene and provide detailed descriptions of them, including their attributes and significance within the context of the panel.

- **Scene Composition**: Describe how the elements are arranged in the scene. Explain what is in the foreground and what is in the background. Note the spatial relationships between characters and objects, and how the composition contributes to the overall narrative or mood.

- **Mood and Atmosphere**: Describe the mood of the scene, including notes on how the visuals contribute to it. Use the following taxonomy, returning only the name in your answer:

  {"moods":{"Positive":[{"name":"Happy","description":"Feeling joyful, content, or delighted."},{"name":"Excited","description":"Feeling enthusiastic, energetic, or eager."},{"name":"Calm","description":"Feeling peaceful, relaxed, or serene."},{"name":"Grateful","description":"Feeling appreciative or thankful."},{"name":"Proud","description":"Feeling satisfied with one's achievements or the achievements of others."}],"Negative":[{"name":"Sad","description":"Feeling down, unhappy, or sorrowful."},{"name":"Angry","description":"Feeling irritated, frustrated, or furious."},{"name":"Anxious","description":"Feeling nervous, worried, or uneasy."},{"name":"Lonely","description":"Feeling isolated, disconnected, or abandoned."},{"name":"Bored","description":"Feeling uninterested, disengaged, or restless."}],"Neutral":[{"name":"Indifferent","description":"Feeling neither particularly positive nor negative."},{"name":"Content","description":"Feeling satisfied but not overly excited."},{"name":"Curious","description":"Feeling interested or inquisitive without strong emotion."},{"name":"Confused","description":"Feeling uncertain or unclear but without strong negative feelings."},{"name":"Pensive","description":"Feeling thoughtful or reflective without strong emotional engagement."}]}}

- **Narrative Elements**: Provide any narrative context or plot development indicated within the panel. Describe any thematic elements or subtexts that contribute to the richness and depth of the content.

- **Q&A**: Generate a list of 5 questions and answers about the panel that focus on fine details (objects and actions), overall story reasoning, and mood. Focus on aspects that are captured visually and may be difficult to get without careful observation.
"""

llama3_1_prompt = """
You are an assistant that extracts objects and attributes from text and outputs it in a list format.
The text is a detailed description of a comic book panel according to the following schema:
{
    "characterList": [ ... ],
    "elementsInScene": [ ... ],
    "actions": [ ... ],
    "importantObjects": [ ... ],
    "sceneComposition": { ... },
    "moodAndAtmosphere": { ... },
    "narrativeElements": { ... },
    "qAndA": [ ... ]
}

We want to extract the list of objects and attributes from the text. The output should be a list of strings, each string representing an object with its attributes.
Usually, the important objects and their attributes are mentioned in:
- characterList, mentioned the characters and their attributes
- elementsInScene, mentioned the objects in the scene and their attributes
- importantObjects, mentioned the important objects and their attributes

Lastly, for every object, provide a list of 3 synonyms, based on the context of the panel. The output should be in the following format:
```list
<object>;<synonym 1>;<synonym 2>;<synonym 3>
...
```
"""

llama3_1_prompt_v2 = """
You are an assistant that extracts objects and attributes from text and outputs it in a list format.
The text is a detailed description of a comic book panel according to the following schema:
{
    "characterList": [ ... ],
    "elementsInScene": [ ... ],
    "actions": [ ... ],
    "importantObjects": [ ... ],
    "sceneComposition": { ... },
    "moodAndAtmosphere": { ... },
    "narrativeElements": { ... },
    "qAndA": [ ... ]
}

Your task is to compact all these information into a single caption that describe the comic panel.
In particular, you should focus on the important objects and their attributes, the scene composition, as well as the characters and their actions.

Once you have a detailed caption, extract the list of objects and their attributes from the caption.
Every object should have 3 synonyms, based on the context of the panel. 
The synonyms should be designed so that substituing the object with its synonyms in the caption still makes sense.

The output should be in the form of:
```text
<caption of the comic panel>
```

```csv
<object>;<synonym 1>;<synonym 2>;<synonym 3>
...
```
"""
