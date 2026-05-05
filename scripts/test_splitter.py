from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "H1"),   # Captures whatever follows # as metadata['H1']
    ("##", "H2"),  # Captures whatever follows ## as metadata['H2']
    ("###", "H3")  # Captures whatever follows ### as metadata['H3']
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    

test_doc = """# Dagger
*Weapon* 
!

- **Damage**: 1d4 piercing
- **Range**: 20/60
- **Properties**: #Finesse

When making an attack with a Finesse weapon, use your choice of your Strength or Dexterity modifier for the attack and damage rolls. You must use the same modifier for both rolls., ## Light

While you hold this weapon, it sheds Bright Light in a 20-foot radius and Dim Light for an additional 20 feet.

*Source: Dungeon Master's Guide (2024) p. 276. Available in the SRD and the Free Rules (2024)* | Required | 10 | 10,000 gp |
| ## Necklace of prayer beads
, #Thrown

If a weapon has the Thrown property, you can throw the weapon to make a ranged attack, and you can draw that weapon as part of the attack. If the weapon is a Melee weapon, use the same ability modifier for the attack and damage rolls that you use for a melee attack with that weapon.
- **Cost**: 2 gp
- **Weight**: 1.0 lbs.

*Source: Player's Handbook (2024) p. 215. Available in the SRD and the Free Rules (2024)*, ## Arcane Focus

# Arcane Focus
*Spellcasting focus* 



**Items in this group:**

- Crystal
- Orb
- Rod
- Staff
- Wand

*Source: Player's Handbook (2024) p. 224* (## Quarterstaff

# Quarterstaff
*Weapon* 
!

- **Damage**:
 - One-handed: 1d6 bludgeoning
 - Two-handed: 1d8 bludgeoning
- **Properties**: Versatile

You gain an Origin feat of your choice.
- **Cost**: 2 sp
- **Weight**: 4.0 lbs.

*Source: Player's Handbook (2024) p. 215. Available in the SRD and the Free Rules (2024)*), ## Robe

# Robe
*Adventuring gear* 


- **Cost**: 1 gp
- **Weight**: 4.0 lbs.

A Robe has vocational or ceremonial significance. Some events and locations admit only people wearing a Robe bearing certain colors or symbols.

*Source: Player's Handbook (2024) p. 228. Available in the SRD and the Free Rules (2024)*, Spellbook, ## Scholar's Pack

# Scholar's Pack
*Adventuring gear* 


- **Cost**: 40 gp
- **Weight**: 22.0 lbs.

A Scholar's Pack contains the following items: Backpack, Book, Ink, Ink Pen, Lamp, 10 flasks of Oil, 10 sheets of Parchment, and Tinderbox.

*Source: Player's Handbook (2024) p. 228. Available in the SRD and the Free Rules (2024)*, and 5 GP; or (B) 55 GP


## Wizard

Wizards are defined by their exhaustive study of magic's inner workings. They cast spells of explosive fire, arcing lightning, subtle deception, and spectacular transformations. Their magic conjures monsters from other planes of existence, glimpses the future, or forms protective barriers. Their mightiest spells change one substance into another, call meteors from the sky, or open portals to other worlds.

Most Wizards share a scholarly approach to magic. They examine the theoretical underpinnings of magic, particularly the categorization of spells into schools of magic. Renowned Wizards such as Bigby, Tasha, Mordenkainen, and Yolande have built on their studies to invent iconic spells now used across the multiverse.

The closest a Wizard is likely to come to an ordinary life is working as a sage or lecturer. Other Wizards sell their services as advisers, serve in military forces, or pursue lives of crime or domination.

But the lure of knowledge calls even the most unadventurous Wizards from the safety of their libraries and laboratories and into crumbling ruins and lost cities. Most Wizards believe that their counterparts in ancient civilizations knew secrets of magic that have been lost to the ages, and discovering those secrets could unlock the path to a power greater than any magic available in the present age.

## Class Features

Spellcasting (Level 1)

As a student of arcane magic, you have learned to cast spells. See "chapter 7" for the rules on spellcasting. The information below details how you use those rules with Wizard spells, which appear in the Wizard spell list later in the class's description.

**Cantrips.** You know three Wizard cantrips of your choice. ## Light

# Light
*cantrip, Evocation* 


- **Casting time:** 1 Action
- **Range:** Touch
- **Components:** V, M (a firefly or phosphorescent moss)
- **Duration:** 1 hour

You touch one Large or smaller object that isn't being worn or carried by someone else. Until the spell ends, the object sheds Bright Light in a 20-foot radius and Dim Light for an additional 20 feet. The light can be colored as you like.

Covering the object with something opaque blocks the light. The spell ends if you cast it again.

**Classes**: Artificer; Bard (College of Lore); Bard; Cleric (Arcana Domain); Cleric; Fighter (Eldritch Knight); Rogue (Arcane Trickster); Sorcerer (Divine Soul); Sorcerer; Warlock (Celestial Patron); Wizard (Evoker); Wizard

*Source: Player's Handbook (2024) p. 292. Available in the SRD and the Free Rules (2024)*, ## Mage Hand

# Mage Hand
*cantrip, Conjuration* 


- **Casting time:** 1 Action
- **Range:** 30 feet
- **Components:** V, S
- **Duration:** 1 minute

A spectral, floating hand appears at a point you choose within range. The hand lasts for the duration. The hand vanishes if it is ever more than 30 feet away from you or if you cast this spell again.

When you cast the spell, you can use the hand to manipulate an object, open an unlocked door or container, stow or retrieve an item from an open container, or pour the contents out of a vial.

As a ## Magic

The Ring of Winter has 12 charges and regains all its expended charges daily at dawn. While wearing the ring, you can expend the necessary number of charges to activate one of the following properties:

- You can expend 1 charge as an action and use the ring to lower the temperature in a 120-foot-radius sphere centered on a point you can see within 300 feet of you. The temperature in that area drops 20 degrees per minute, to a minimum of -30 degrees Fahrenheit. Frost and ice begin to form on surfaces once the temperature drops below 32 degrees. This effect is permanent unless you use the ring to end it as an action, at which point the temperature in the area returns to normal at a rate of 10 degrees per minute. 
- You can cast one of the following spells from the ring (spell save DC 17) by expending the necessary number of charges: ## Bigby's hand
 action on your later turns, you can control the hand thus again. As part of that action, you can move the hand up to 30 feet."""
splits = markdown_splitter.split_text(test_doc)

for i, chunk in enumerate(splits):
    print(f"--- Chunk {i} Metadata ---")
    print(chunk.metadata)
