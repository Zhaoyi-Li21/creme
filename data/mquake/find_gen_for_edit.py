import json
import random
random.seed(0)
dir_path = "/root/autodl-tmp/zhaoyi" # your dictionary path
read_path = dir_path + "/creme/data/mquake/MQuAKE-CF-3k.2hop.json"
write_path = dir_path + "/creme/data/mquake/MQuAKE-CF-3k.2hop.edit.json"
fr = open(read_path, "r")
fw = open(write_path, "w")
data = json.load(fr)

hop_1_file = dir_path + "/creme/data/mquake/prompts/templates/rewrite_template_for_hop1.json"
hop_2_file = dir_path + "/creme/data/mquake/prompts/templates/rewrite_template_for_hop2.json"
question_file = dir_path + "/creme/data/mquake/prompts/templates/question_templates.json"
fr_question = open(question_file, "r")
question_dict = json.load(fr_question)
hop_1_fr = open(hop_1_file, "r")
hop_2_fr = open(hop_2_file, "r")
hop_1_temp_dict = json.load(hop_1_fr)
hop_2_temp_dict = json.load(hop_2_fr)

output_data = list()

gen_number = 5 # find how many generalization cases
irre_number = 5 # find how many generalization cases
for i in range(len(data)):
    output_datum = dict()
    datum = data[i]
    output_datum["questions"] = datum["questions"]
    first_hop_result = datum["orig"]["triples"][0][2]
    second_hop_result = datum["orig"]["triples"][1][2]
    second_hop_rel = datum["orig"]["triples"][1][1]
    candidates = list()
    candidates_set = list()
    for j in range(len(data)):
        if j == i:
            continue
        if data[j]["orig"]["triples"][0][2] == first_hop_result:
            if data[j]["orig"]["triples"][1][2] == second_hop_result:
                continue # find different second hops
            if data[j]["orig"]["triples"][1][2] in candidates_set:
                continue # deduplicate
            candidates_set.append(data[j]["orig"]["triples"][1][2])
            candidates.append((data[j]["orig"]["triples"][1], 
                               data[j]["orig"]["triples_labeled"][1]))
        
    if len(candidates) > gen_number:
        gens = random.sample(candidates, gen_number)
    else:
        gens = candidates

    output_datum["generalization clozes"]=list()
    output_datum["generalization questions"]=list()
    output_datum["generalization answers"]=list()
    for gen in gens:
        gen_triple, gen_triple_labeled = gen
        triple_1_code = datum["orig"]["triples"][0][1]
        hop_1_temp = hop_1_temp_dict[triple_1_code]
        gen_triple_code = gen_triple[1]
        gen_cloze_temp = hop_2_temp_dict[gen_triple_code]
        gen_question_temp = question_dict[gen_triple_code]
        gen_answer = gen_triple_labeled[2]

        triple_1_s = datum["orig"]["triples_labeled"][0][0]

        hop_1_str = hop_1_temp.replace("[X]", triple_1_s)
        gen_cloze = gen_cloze_temp.replace("[X]", hop_1_str)
        gen_question = gen_question_temp.replace("[X]", hop_1_str)

        output_datum["generalization clozes"].append(gen_cloze.split(' __')[0])
        output_datum["generalization questions"].append(gen_question)
        output_datum["generalization answers"].append(gen_answer)

    candidates = list()
    candidates_set = list()
    for j in range(len(data)):
        if j == i:
            continue
        if data[j]["orig"]["triples"][1][1] == second_hop_rel:
            if data[j]["orig"]["triples"][0][2] == first_hop_result:
                continue

            if data[j]["orig"]["triples"][0][2] in candidates_set:
                continue # deduplicate
            candidates_set.append(data[j]["orig"]["triples"][0][2])
            candidates.append((data[j]["orig"]["triples"], 
                               data[j]["orig"]["triples_labeled"]))

    if len(candidates) > irre_number:
        irres = random.sample(candidates, irre_number)
    else:
        irres = candidates

    output_datum["irrelevant clozes"]=list()
    output_datum["irrelevant questions"]=list()
    output_datum["irrelevant answers"]=list()
    for irre in irres:
        irre_triples, irre_triples_labeled = irre
        triple_1_code = irre_triples[0][1]
        hop_1_temp = hop_1_temp_dict[triple_1_code]
        triple_2_code = irre_triples[1][1]
        hop_2_temp = hop_2_temp_dict[triple_2_code]
        irre_question_temp = question_dict[triple_2_code]
        irre_answer = irre_triples_labeled[1][2]

        triple_1_s = irre_triples_labeled[0][0]

        hop_1_str = hop_1_temp.replace("[X]", triple_1_s)
        irre_cloze = hop_2_temp.replace("[X]", hop_1_str)
        irre_question = irre_question_temp.replace("[X]", hop_1_str)

        output_datum["irrelevant clozes"].append(irre_cloze.split(' __')[0])
        output_datum["irrelevant questions"].append(irre_question)
        output_datum["irrelevant answers"].append(irre_answer)
    output_data.append(output_datum)
json.dump(output_data,fw,indent=4)


            