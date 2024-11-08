# Annotation Format Specification

## Overview
This document outlines the XML-based annotation format for the Manga109 dataset and similar datasets. The format is designed to represent various annotations, including characters, speech balloons, panels, and other elements typically found in manga or similar graphic novels.

## Structure

### Root Element
- `<annotations>`: The root element of the document.

### Book Element
- `<book>`: Represents a single book or volume.
    - **Attributes**:
        - `title`: The title of the book.
    - **Content**:
        - `<characters>`: A list of characters in the book.
        - `<stories>`: A collection of stories elements.
        - `<pages>`: A collection of page elements.

### Characters Element
- `<characters>`: Contains a list of all characters present in the book.
    - **Content**:
        - `<character>`: Individual character element.
            - **Attributes**:
                - `id`: A unique identifier for the character.
                - `name`: The name of the character.
      
### Advertisement
- `<adds>`: Contains a list of all adds present in the book.
    - **Content**:
        - `<advertisement>`: Individual advertisement element.
            - **Attributes**:
                - `id`: A unique identifier for the add.
                - `name`: The name of the add.
                - `type`: The type of add (e.g., "promotion", "editorial")

### Stories Element
- `<stories>`: Contains a list of all stories present in the book.
    - **Content**:
        - `<story>`: Individual story element.
            - **Attributes**:
                - `id`: A unique identifier for the story.
                - `name`: The name of the story.

### Page Element
- `<page>`: Represents a single page in the book.
    - **Attributes**:
        - `id`: The page number or index.
        - `width`: Width of the page in pixels.
        - `height`: Height of the page in pixels.
        - `type`: The type of page (e.g., "color", "monochrome").
        - `story_id` (optional): ID of the story associated with this page.
        - `add_id` (optional): ID of the addvertisement within this page.
    - **Content**:
        - Annotation elements (`<panel>`, `<text>`, `<character>`, `<balloon>`, etc.).

### Annotation Elements
- `<panel>`: (polygon with 4 points) Represent a Panel within the page.
    - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".

- `<text>`: (polygon with 4 points) Used for annotating text within the page.
    - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".
        - `balloon_id` (optional): ID of the balloon associated with this annotation.

- `<character>`: (polygon with 4 points) Used for annotating character within the page.
    - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".
        - `character_id` (optional): ID of the character associated with this annotation.

- `<balloon>`: (polygon with N points) Used for annotating speech balloons within the page.
    - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".

- `<face>`: Used for annotating character faces within the page.
    - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".
        - `character_id` (optional): ID of the character associated with this annotation.

- `<onomatopoeia>`: Used for annotating panels within the page.
  - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".

- `<link_sbsc>`: (polyline with 2 points) Links between text elements and characters, within the page.
  - **Attributes**:
        - `id`: A unique identifier for the annotation.
        - `points`: A series of x,y coordinates in the format "x0,y0 x1,y1 ...".
        - `character_id` (optional): ID of the body associated with this annotation.
        - `text_id` (optional): ID of the text associated with this annotation.

Attributes within elements can include additional descriptive information relevant to the specific annotation.

## Usage

This format is intended for use in datasets where detailed annotations of graphic novels, manga, or similar materials are required. It supports a wide range of annotations, from character identification to speech balloons, and allows for complex relationships between different elements on a page.