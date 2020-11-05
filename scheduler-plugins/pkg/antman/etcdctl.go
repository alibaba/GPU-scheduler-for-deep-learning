package antman

import (
	"os"
	"os/exec"
	"strings"

	"k8s.io/klog"
)

type EtcdWrapper struct {
	basic_etcd_cmd string
	domain         string
}

func NewEtcdWrapper() *EtcdWrapper {
	api_server_ip := "192.168.1.57"
	api_server_port := "2379"
	pem_prefix := "/etc/kubernetes/pki/etcd"

	var cmd string
	cmd = "etcdctl --endpoints=https://" + api_server_ip + ":" + api_server_port + " "
	cmd += "--cacert=" + pem_prefix + "/ca.pem "
	cmd += "--cert=" + pem_prefix + "/etcd-client.pem "
	cmd += "--key=" + pem_prefix + "/etcd-client-key.pem "

	os.Setenv("ETCDCTL_API", "3")

	ew := &EtcdWrapper{
		basic_etcd_cmd: cmd,
		domain:         "/antman",
	}
	return ew
}

func (ew EtcdWrapper) ReadEtcd(key *string) *string {
	cmd := ew.basic_etcd_cmd
	cmd += "get "
	cmd += ew.domain + "/" + *key + " "

	cmds := strings.Split(cmd, " ")

	out, err := exec.Command(cmds[0], cmds[1:]...).CombinedOutput()
	if err != nil {
		klog.Errorf("cmd.Run() failed with %v\n", err)
		return nil
	}

	outstr := string(out)

	tokens := strings.Split(outstr, "\n")

	// klog.Infof("combined out:\n%s\n", tokens[1])
	if len(tokens) < 2 {
		// not an expected result, treated as empty key/val
		return nil
	}

	// the second one is the value
	return &tokens[1]
}

func (ew EtcdWrapper) WriteEtcd(key *string, val *string) {
	cmd := ew.basic_etcd_cmd
	cmd += "put "
	cmd += ew.domain + "/" + *key + " " + *val

	cmds := strings.Split(cmd, " ")

	_, err := exec.Command(cmds[0], cmds[1:]...).CombinedOutput()
	if err != nil {
		klog.Errorf("cmd.Run() failed with %v\n", err)
	}

	return
}
