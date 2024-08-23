text = "123456789012345678901234567890"
chunk_size = 10
overlap =  3

chunks = [
    text[i: i+ chunk_size]
    for i in range(0, len(text), chunk_size-overlap)
]

print(chunks)