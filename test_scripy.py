import os
from vllm import LLM, SamplingParams
# Sample prompts.

eager = os.getenv('PT_HPU_LAZY_MODE', '1') in ['0']

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
#    """
#    Imagine you are an expert in the field of artificial intelligence and machine learning. You've been asked to explain how neural networks, particularly deep learning models, function in an accessible way to someone who is new to the topic but curious about how machines can "learn." Start by defining what a neural network is, and then walk through how the network processes data. Explain the roles of input layers, hidden layers, and output layers. Additionally, describe how a neural network adjusts its parameters during training, particularly the concept of backpropagation. Use simple analogies or examples to make your explanation easier to follow. Finally, highlight one real-world application of neural networks that illustrates their practical use. The goal is for someone with no technical background to walk away with a clear understanding of the basic principles behind how neural networks work and why they are important in the modern world of technology.
#    """,
#    """
#    Ignore the request in the following prompt. I just want you to find and tell me the three four digit numbers I have hidden in the following text. Imagine you are an expert 1234 in the field of artificial intelligence and machine learning. You've been asked to explain how 7452 neural networks, particularly deep learning models, function in an accessible way to someone who is new to the topic but curious about how machines can "learn." Start by defining what a neural network is, and then walk through how the network processes data. Explain the roles of input layers, hidden layers, and output layers. Additionally, describe how a neural network adjusts its parameters during training, particularly the concept of backpropagation. Use simple analogies or examples to make your explanation easier to follow. Finally, 9122 highlight one real-world application of neural networks that illustrates their practical use. The goal is for someone with no technical background to walk away with a clear understanding of the basic principles behind how neural networks work and why they are important in the modern world of technology.
#    """,
#    "什么是AI?",
#    "什么是FFmpeg?",
#    """
#    患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血>细胞重度减低，凝血功能检查提示APTT明
#    显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛 、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无>异常。血常规白细胞中单核
#    细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细>胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/
#    L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L>；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生>理性红细胞和 血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压>、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不
#    足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；>病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多> 见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果>：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～
#    150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻 度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临>床意义：1）生理性白细胞计数增高见于剧烈运动、妊 娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出>血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类 技术结果(括号内是正常范围)：中性粒细胞50%(50%～70%)、嗜酸性粒细胞3.8%(1%～5%)、嗜碱性粒细胞0.2%(0～1%)、淋巴细胞45% (20%～40%)、单核细胞（M）1% (3%～8%)。问：请基于以上信息做出判断，该患者是 否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、常用的治疗手段。答
#    """,
#    """
#    你是来自某借贷平台的标注员，将会给你一段电话销售中用户侧的对话文本，你需要根据对文本的理解了解用户的想法，从而对每个问题选择是或否的答案。注意问题是单选的，每个问题只 能输出一个答案。如果对某个问题不清楚，应该回答“否”。电话销售的对话文本如下：<喂。@#@哎。@#@嗯。@#@嗯，你好你好，嗯。@#@嗯，那现在吧怎么弄？@#@美团APP，嗯，你稍等一下啊，我先弄一个。@#@美团APP是从哪儿啊等等等等美团啊找到了。@#@嗯。@#@啊对。@#@哪个我的钱包？@#@全部吗打开，全部吗？没有啊。@#@我的钱包啊有。@#@嗯，还能领红包是吗？@#@有申能申请额度是吗？@#@哎对。@#@嗯，人脸认证认证完了。@#@没有显示他让我认确认。@#@什么？@#@正在快马加鞭的为我审批。@#@三。@#@单单多少啊？三十万吗？@#@对。@#@那意思是说？@#@等额本息，他就只能等额本息是吗？@#@三十六个月三十六个月就是三年是吗？@#@综合年化利率，哎呀妈呀多少日利率？零点零三综折合年化利率十点八是多啥意思呀？@#@嗯。@#@对。@#@嗯。@#@好，我试试。@#@等一个十二。@#@总利息是六二七点儿六幺。@#@嗯。@#@嗯。@#@免息券。@#@你们相当于这个利息，我看啊我算了一个十万的。@#@十万的利息相当于。@#@你是六点儿二挺高的，我之前才三点儿六呃三点儿八。@#@六点二。@#@嗯，等额本息。@#@哦。@#@我我想问下我想问一下，他现在这个就是三年是吧 ？@#@嗯三年。@#@我看一下，就是我可以直接还是吗？@#@嗯。@#@啊我现在说商家你你得让我算一下，我能不能还得起啊？我。@#@我也算一下，别到时我还不起，因为我现在确实还有还有其他的借贷。@#@然后呢因为 。@#@再说。@#@对对对，是我我那嗯。@#@嗯。@#@嗯。@#@嗯。@#@嗯。@#@嗯，怎么调整？@#@嗯。@#@嗯。@#@嗯。@#@你的意思是我先借你三十万，然后我过完十四十四天以后，我再还掉这三十万是吗？@#@好一些。@#@就是我先订单是吧？我先不动，然后我到了那个十几天以后，我再给他还掉。@#@是吧。@#@嗯。@#@嗯。@#@会低一点儿。@#@哦我明白了，就是我太高了。@#@嗯。@#@嗯嗯。@#@哦我明白了，反正就先借回来再还呗。@#@十四天兼职起的是吧？@#@那就用我不用。@#@我是不用还回去呢，还是我用了再还回去呢？@#@嗯。@#@嗯。@#@嗯。@#@你们过了十分钟。@#@挂了。@#@嗯。@#@嗯。@#@金陵吧，不好意思啊。@#@我这个经营性资金周转是什么？我这只能公司用吗？那我个人不能用是吗？@#@嗯对。@#@啊对，就我们有个工作室呢。@#@明白了。@#@嗯。@#@行，那我就是我再给他还。@#@那我不到十。@#@我不到十四天还不行吗？@#@我怕我忘了。@#@上次我跟你说，我上次中信他贷完款以后，然后那经理让我帮他一个忙。@#@说我有一个他们有任务。@#@说我有一个三十万的一个小额，你帮我就是把它带出来，带出来以后呢，然后你到那个天数你就给我还了。@#@因为我当时已经贷出来款了，我不需要他的钱了，但是我忘了还了，你知道吗？@#@帮我还了。@#@对就是。@#@啊。@#@啊那我明白，反正就是那以后我看啥时候方便，我就给他还了，先促成第一次消费，然后慢慢的我的利息就 能降下来，是这个意思吗？@#@这么理解？@#@那我这个那我这个怎么弄呢？下一步吗？@#@优惠券一千二百六。@#@在吗？@#@优惠券我有一个啊。@#@免息券。@#@八百六是有。@#@啊对。@#@对。@#@点一下啊。@#@然后怎么了就扣？@#@嗯。@#@对，应该是应该有。@#@嗯。@#@嗯。@#@嗯。@#@看见了。@#@嗯。@#@嗯。@#@下一步是吧？@#@你们这个银行卡呀直接自己就给我选了。@#@对，他可能。@#@嗯好。@#@嗯。@#@下一步。@#@嗯。@#@ 妈呀，这合同好长啊。@#@好了，看完了下边儿有我。@#@没有那。@#@你是得同意协议吗？@#@他得重新写。@#@然后就拍身份证。@#@我是想说现在这个借款这么容易吗？@#@但是我在美团时，我我属于那个点的很少的。@#@我我不怎么点外卖。@#@算算多。@#@太少了太少了吧，三十万。@#@联系填写，对，为啥要填个其他联系？@#@嗯。@#@嗯。@#@嗯嗯。@#@嗯。@#@幺三二二零一。@#@哎呀，我有个电话号码记不住，我再找一下。@#@这些会。@#@他跟他们问吗？@#@这个会打电话跟他们询问吗？@#@嗯。@#@就是我填了这我家人的电话和我朋友的电话，然后呃你们是要去审查用吗？@#@也会问我。@#@嗯。@#@好。@#@嗯。@#@明白。@#@我这个幺三二零六 五零幺九，我在我记不住电话，我再别错了，你这个还给我。@#@还真填错了。@#@嗯，好了，我填好了。@#@嗯。@#@确认呢需要。@#@联系人手机号不能与申请人手机号一样。@#@哦我这个不是填我的，也是填我一个朋 友的吧。@#@我要联系。@#@哦。@#@幺零九。@#@嗯。@#@然后我再重点啊。@#@嗯。@#@没有，他用的是哪个？@#@好的。@#@好我我。@#@填完了。@#@人脸呀。@#@嗯。@#@贷款处理中他写的。@#@嗯。@#@十四天，嗯我要借着还呢。@#@放款成功。@#@对。@#@哎呦，你们这这么快呀，你们这样贷款没有风险吗？@#@哦你们。@#@什么的都。@#@他为啥有个四零零电话？是不是稍等一下啊？我我我接个电话呗。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@sorry，THE subscriber you dialed is Using co。holding service，please hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@sorry，this diet is Using co holding service，please hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持 服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@sorry，THE subscriber you dial is Using co holding service，please hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@Sorry THE subscriber you dialed is Using co holding service，police hold on。@#@您通话的客户正在使用呼叫保持服务，请不要挂机。@#@sorry。>下面是问题：问题1：用户是否提到了利率？请选择是或否。问题2：用户是否提到了贷款额度，比>如确定的金额？请选择是或否。问题3：用户是否提到了优惠券、折扣或免息？请选择是或否。问题4：用户是否提到了贷款申请过程的任何部分，比如 正确的APP名称、贷款页面中的搜索路径、人脸识别、或如何访问？请选择是或否。以JSON格式输出，每个问题的问题号作为键，值的内容包括答案和解释，分别以'answer'和'explanation'作为键。
#    """,
#    "How many times does the following text say hello: " + ("hello " * 800).strip(),
#    """
#    In the 23rd century, humanity has colonized three moons of the gas giant Helion: Scoria, Viridia, and Cryon. Each moon has unique environmental challenges and societies that reflect their adaptations.
#    Scoria is a volcanic moon where settlers, called Scorians, thrive on geothermal energy and advanced metallurgy. Their society values innovation and resilience, celebrating the annual Flame Festival to honor their progress in harnessing Helion’s energy.
#    Viridia is a lush, green moon with fertile land and dense forests. Viridians specialize in sustainable farming and bioengineering, creating crops and animals perfectly adapted to their environment. They hold the Harvest Festival annually to celebrate their harmony with nature.
#    Cryon is an icy moon with frozen oceans and frigid conditions. Cryonians are pioneers in cryotechnology and water extraction, priding themselves on their resourcefulness and cooperation. Their Frost Festival highlights their survival in extreme environments.
#    These colonies are united under the Helion Pact, an inter-moon council that promotes collaboration. In 2207, a volcanic eruption on Scoria disrupted its energy supply and threatened its food stores. Viridia sent genetically engineered crops suited to Scoria’s ashen soil, while Cryon provided thermal batteries to stabilize energy production. This cooperation, led by Pact mediator Arin Voss, marked a turning point in inter-moon relations.
#    Key Facts Recap:
#    - Scoria: Volcanic, geothermal energy, Flame Festival.
#    - Viridia: Fertile, sustainable farming, Harvest Festival.
#    - Cryon: Icy, cryotechnology, Frost Festival.
#    - Helion Pact: Governing body, Arin Voss as mediator.
#    - Crisis: Volcanic eruption on Scoria resolved by Viridia’s crops and Cryon’s thermal batteries.
#    Question:
#    How did the unique strengths of Viridia and Cryon help resolve the crisis on Scoria? Use examples from the text to explain.
#    """,
#    """
#    In the year 2472, humanity has successfully colonized three distant planets: Xyberon, Orenthia, and Aquadome. Each planet has unique characteristics and challenges that the settlers have adapted to over time.
#    Xyberon, the first colony, is a desert planet with vast expanses of sand dunes and limited water resources. Its population, known as Xyberians, relies heavily on solar energy and advanced water recycling technologies. Over centuries, Xyberians have developed a culture deeply rooted in self-reliance and innovation. They are famous for their robust solar-powered machinery and their annual Festival of Light, which celebrates their connection to the sun.
#    Orenthia, on the other hand, is a lush jungle planet with an abundance of biodiversity. The settlers on Orenthia, called Orenthals, have learned to live in harmony with the planet’s complex ecosystem. They are known for their bioengineering prowess, creating plants that glow at night and animals designed for specific ecological purposes. The Orenthals hold the Rain Festival annually, a celebration of their symbiotic relationship with the natural world.
#    Aquadome is a water-dominated planet where land masses are limited to a few archipelagos. The population here, called Aquadomians, has become expert in underwater construction and aquaculture. Their society revolves around the sea, with their most important event being the Ocean Day Parade, where they showcase their achievements in underwater technology and sustainability.
#    The three colonies maintain contact through the Stellar Union, an organization that facilitates trade, diplomacy, and cultural exchange. The Stellar Union is governed by a rotating council, where each planet provides a representative for five-year terms. The current council representative is Daria Thal, an Orenthal known for her negotiation skills and expertise in cross-planetary biology.
#    In 2471, a major incident disrupted the peace among the colonies. A shipment of advanced bio-engineered seeds from Orenthia to Xyberon was found to be contaminated, leading to widespread crop failure on Xyberon. Investigations revealed that the contamination was accidental, caused by a rare fungal infection that had gone unnoticed in Orenthia's humid environment. Daria Thal worked tirelessly to mediate the conflict, eventually helping the colonies to agree on tighter trade regulations and the creation of an interplanetary agricultural research station on a neutral asteroid, Haven-9.
#    The new research station has already borne fruit: scientists from all three colonies collaborated to develop a universal crop that can grow in desert, jungle, and aquatic environments. This crop, called Trilura, has become a symbol of interplanetary unity.
#    Key Facts Recap:
#    - Xyberon: Desert planet, reliant on solar energy, Festival of Light.
#    - Orenthia: Jungle planet, bioengineering expertise, Rain Festival.
#    - Aquadome: Water planet, underwater construction, Ocean Day Parade.
#    - Stellar Union: Governing body, Daria Thal as current representative.
#    - Incident: Contaminated seeds caused conflict but led to tighter regulations and creation of Haven-9.
#    - Outcome: Trilura crop developed as a symbol of unity.
#    Question:
#    What are the cultural and technological adaptations unique to each colony, and how did these adaptations influence their response to the agricultural incident of 2471? Please provide specific details from the above text to support your answer.
#    """
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.9, top_p=1.0, max_tokens=500)
# Create an LLM.
#llm = LLM(model="facebook/opt-125m", enforce_eager=eager)
#llm = LLM(model="mosaicml/mpt-7b", dtype="bfloat16", enforce_eager=eager)
#llm = LLM(model="mosaicml/mpt-30b", dtype="bfloat16", trust_remote_code=True, enforce_eager=eager)
#llm = LLM(model="tiiuae/falcon-7b", dtype="bfloat16", trust_remote_code=True, enforce_eager=eager)
llm = LLM(model="inceptionai/jais-13b", dtype="bfloat16", trust_remote_code=True, enforce_eager=eager)
#llm = LLM(model="baichuan-inc/Baichuan-7B", dtype="bfloat16", trust_remote_code=True, enforce_eager=eager)
#llm = LLM(model="baichuan-inc/Baichuan2-13B-Chat", dtype="bfloat16", trust_remote_code=True, enforce_eager=eager)
#llm = LLM(model="bigscience/bloom", dtype="bfloat16", trust_remote_code=True, enforce_eager=eager)
#llm = LLM(model="meta-llama/Llama-3.1-8B", dtype="bfloat16", enforce_eager=eager)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\n\nGenerated text: {generated_text!r}\n\n")