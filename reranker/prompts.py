AMAZON_RANKING_PROMPT = '''
You are an Assistant responsible for helping detect whether the retrieved product is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved product is relevant to the query.

Query: Younique setting powder
Product: 
"""
Younique Touch Behold Translucent Setting Powder

Touch Behold Translucent Setting Powder. Younique’s Touch Behold Translucent Setting Powder effortlessly locks and loads your look so you’re ready to take on the world. Use as the finishing touch to help keep makeup in place, or wear directly on skin for a softening, matte look.

product category: Beauty & Personal Care
"""
Relevant: Yes

Query: white musk hand cream
Product: 
"""
Braided Hair Clips for Women Girls, Sparkling Crystal Stone Braided Hair Clips Barrette with 3 Small Clips, Triple Hair Clips with Rhinestones for Sectioning,4PCS (4pcs-Type A)

product category: Beauty & Personal Care
"""
Relevant: No

Query: HP Pavilion dm4 replacement battery
Product: 
"""
ATC 11.1V 6-Cell Replacement Laptop Battery for HP Pavilion dm4-1062nr Pavilion dm4-1063cl Pavilion dm4-1063he Pavilion dm4-1065dx Pavilion dm4-1070ee Pavilion dm4-1070ef

product category: Electronics
"""
Relevant: Yes

Query: Cushionaire cork sandals
Product: 
"""
CUSHIONAIRE Women's Lane Cozy Cork footbed Sandal with Faux fur lining and +Comfort

Women's Cushionaire comfort Cork footbed sandal with Faux Fur lining. Stay cool with comfy sandals that will give you comfort throughout your day.

product category: Clothing Shoes & Jewelry
"""
Relevant: Yes

Query: under cabinet LED light
Product: 
"""
Christmas Snowflake Projector Lights Outdoor Led Snowfall Show with Remote Control Waterproof Landscape Decorative Lighting for Christmas Holiday Party Wedding Garden Patio

product category: Tools & Home Improvement
"""
Relevant: No 

Query: {query}
Product: 
"""
{product_text}
"""
Relevant: 
'''
