from benchflow import BaseAgent
import subprocess
import re
import shutil
import json
import os
import sys

class SWEAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.model_name = "gpt-4o-mini"
        
    def call_api(self) -> str:
        print(self.env_info)
        instance_id = self.env_info["instance_id"]
        print(instance_id)
        shutil.rmtree("trajectories/root/", ignore_errors=True)
        cmd = f"sweagent run-batch --agent.model.name {self.model_name} --instances.split test --instances.filter {instance_id}"

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
        process.stdout.close()
        process.wait()

        return self.parse_action(instance_id, ''.join(output_lines))
        
    def parse_action(self, instance_id: str, log_content: str) -> str:
        pattern = r"Wrote merged predictions to\s+(\S.*?)(?=\s*\n|$)"
        match = re.search(pattern, log_content, re.DOTALL)

        if match:
            file_path = match.group(1).strip()
            if not os.path.exists(file_path):
                print(f"error: {file_path} not exists")
            else:
                try:
                    with open(file_path, 'r') as f:
                        predictions = json.load(f)
                    action = predictions[instance_id]['model_patch']
                    print(action)
                    return action
                except json.JSONDecodeError:
                    print(f"error: {file_path} not a valid json")
                    return None
                except Exception as e:
                    print(f"error: {str(e)}")
                    return None
        else:
            print("error: no predictions file path")
            return None
def main():
    agent = SWEAgent()
    agent.run_with_endpoint(host="0.0.0.0", port=9002)

if __name__ == "__main__":
    main()
