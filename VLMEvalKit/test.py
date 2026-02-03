def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False


def split_MMMU(msgs):
    text, images = None, []
    for s in msgs:
        if s['type'] == 'image':
            images.append(s['value'])
        elif s['type'] == 'text':
            assert text is None
            text = s['value']
    text_segs = text.split('<image ')
    if len(text_segs) == 1:
        return msgs

    segs = [dict(type='text', value=text_segs[0])]
    for i, seg in enumerate(text_segs):
        if i == 0:
            continue
        assert istype(seg[0], int) and seg[1] == '>'
        image_idx = int(seg[0]) - 1
        segs.append(dict(type='image', value=images[image_idx]))
        segs.append(dict(type='text', value=seg[2:]))
    return segs


msg = [
    {
        "type": "text",
        "value": """
            Question:
            In sets a - d, only one of the set is incorrect regarding basic strength. Select it :
            Options:
            A) <image 1>
            B) <image 2>
            C) <image 3>
            D) <image 4>
        """
    },
    {
        "type": "image",
        "value": "path/to/image1.jpg"
    },
    {
        "type": "image",
        "value": "path/to/image2.jpg"
    },
    {
        "type": "image",
        "value": "path/to/image3.jpg"
    },
    {
        "type": "image",
        "value": "path/to/image4.jpg"
    }
]

res = split_MMMU(msg)
print(res)
